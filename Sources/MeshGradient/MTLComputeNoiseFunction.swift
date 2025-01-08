import Foundation
import Metal
import simd
@_implementationOnly import MeshGradientCHeaders

/// Tracks memory usage of Metal resources
struct MetalMemoryTracker {
    static var shared = MetalMemoryTracker()
    private var memoryUsage: [String: Int] = [:]
    private let queue = DispatchQueue(label: "com.meshgradient.memorytracker")
    
    mutating func track(bytes: Int, identifier: String) {
        queue.sync {
            memoryUsage[identifier] = bytes
        }
    }
    
    mutating func untrack(identifier: String) {
        queue.sync {
            memoryUsage.removeValue(forKey: identifier)
        }
    }
    
    func totalMemoryUsage() -> Int {
        queue.sync {
            return memoryUsage.values.reduce(0, +)
        }
    }
    
    func memoryReport() -> String {
        queue.sync {
            return memoryUsage.map { "\($0.key): \(Double($0.value) / 1_000_000.0)MB" }.joined(separator: "\n")
        }
    }
}

final class MTLComputeNoiseFunction {
    private weak var _noiseTexture: MTLTexture?
    private var lastViewportSize: simd_float2 = .zero
    private let device: MTLDevice
    private let pipelineState: MTLComputePipelineState
    private var memoryTracker = MetalMemoryTracker.shared
    private let textureIdentifier = "NoiseTexture"
    
    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let computeNoiseFunction = library.makeFunction(name: "computeNoize")
        else { throw MeshGradientError.metalFunctionNotFound(name: "computeNoize") }
        self.pipelineState = try device.makeComputePipelineState(function: computeNoiseFunction)
    }
    
    deinit {
        purgeTextures()
    }
    
    func purgeTextures() {
        _noiseTexture = nil
        lastViewportSize = .zero
        memoryTracker.untrack(identifier: textureIdentifier)
    }
    
    private func selectOptimalPixelFormat(for device: MTLDevice) -> MTLPixelFormat {
        // Check for best compression support in order of efficiency
        if device.supportsFamily(.apple7) {
            return .bgra8Unorm_srgb // Most efficient on modern Apple Silicon
        } else if device.supportsFamily(.apple6) && device.supportsTexture(descriptor: {
            let desc = MTLTextureDescriptor()
            desc.pixelFormat = .astc_4x4_ldr
            return desc
        }())) {
            return .astc_4x4_ldr // Best compression ratio ~4:1
        } else if device.supportsFamily(.apple3) && device.supportsTexture(descriptor: {
            let desc = MTLTextureDescriptor()
            desc.pixelFormat = .bc7_rgbaUnorm
            return desc
        }())) {
            return .bc7_rgbaUnorm // Good quality compression ~4:1
        } else if device.supportsFamily(.apple3) {
            return .rgb9e5Float // Compressed floating point, good for gradients
        }
        return .bgra8Unorm // Fallback, still relatively efficient
    }
    
    private func optimizeTextureDescriptor(_ descriptor: MTLTextureDescriptor, width: Int, height: Int) {
        // Enable mipmaps for better memory usage when scaling
        descriptor.mipmapLevelCount = max(1, Int(log2(Double(max(width, height)))))
        
        // Set optimal storage mode
        if device.hasUnifiedMemory {
            descriptor.storageMode = .shared
        } else {
            descriptor.storageMode = .private
            if #available(macOS 11.0, iOS 13.0, *) {
                descriptor.hazardTrackingMode = .tracked
            }
        }
        
        // Set optimal cache mode
        if device.hasUnifiedMemory {
            descriptor.resourceOptions = .storageModeShared
        } else {
            descriptor.resourceOptions = .cpuCacheModeWriteCombined
        }
        
        // Enable texture compression
        descriptor.allowGPUOptimizedContents = true
        
        // Set usage for optimal performance
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
    }
    
    func call(viewportSize: simd_float2, pixelFormat: MTLPixelFormat, commandQueue: MTLCommandQueue, uniforms: NoiseUniforms) -> MTLTexture? {
        guard uniforms.noiseAlpha > 0 else { 
            purgeTextures()
            return nil 
        }
        
        // Round viewport size to power of 2 for better compression
        let width = Int(viewportSize.x.rounded(.up))
        let height = Int(viewportSize.y.rounded(.up))
        let roundedWidth = 1 << Int(log2(Double(width)).rounded(.up))
        let roundedHeight = 1 << Int(log2(Double(height)).rounded(.up))
        
        if lastViewportSize == viewportSize, let existingTexture = _noiseTexture {
            return existingTexture
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Clear old texture and its memory tracking
        purgeTextures()
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: selectOptimalPixelFormat(for: device),
            width: roundedWidth,
            height: roundedHeight,
            mipmapped: true
        )
        optimizeTextureDescriptor(textureDescriptor, width: roundedWidth, height: roundedHeight)
        
        guard let noiseTexture = device.makeTexture(descriptor: textureDescriptor),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }
        
        // Track memory usage - estimate based on format and compression
        let bytesPerPixel: Int
        switch noiseTexture.pixelFormat {
        case .astc_4x4_ldr, .bc7_rgbaUnorm:
            bytesPerPixel = 1 // 4:1 compression
        case .rgb9e5Float:
            bytesPerPixel = 2 // 2:1 compression
        case .bgra8Unorm, .bgra8Unorm_srgb:
            bytesPerPixel = 4
        default:
            bytesPerPixel = 4
        }
        
        // Account for mipmaps in memory calculation
        let mipLevels = textureDescriptor.mipmapLevelCount
        let totalPixels = (roundedWidth * roundedHeight * 4) / 3 // Geometric series sum for mipmaps
        let memoryUsage = totalPixels * bytesPerPixel
        memoryTracker.track(bytes: memoryUsage, identifier: textureIdentifier)
        
        let threadgroupCounts = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (roundedWidth + threadgroupCounts.width - 1) / threadgroupCounts.width,
            height: (roundedHeight + threadgroupCounts.height - 1) / threadgroupCounts.height,
            depth: 1
        )
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(noiseTexture, index: Int(ComputeNoiseInputIndexOutputTexture.rawValue))
        
        var uniformsCopy = uniforms
        encoder.setBytes(&uniformsCopy,
                        length: MemoryLayout.size(ofValue: uniformsCopy),
                        index: Int(ComputeNoiseInputIndexUniforms.rawValue))
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupCounts)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        lastViewportSize = viewportSize
        _noiseTexture = noiseTexture
        
        #if DEBUG
        print("Noise Texture Details:")
        print(" - Format: \(noiseTexture.pixelFormat)")
        print(" - Size: \(roundedWidth)x\(roundedHeight)")
        print(" - Mip Levels: \(mipLevels)")
        print(" - Estimated Memory: \(Double(memoryUsage) / 1_000_000.0)MB")
        print("Metal Memory Usage Report:\n\(memoryTracker.memoryReport())")
        #endif
        
        return noiseTexture
    }
}

extension MTLTexture {
    func getPixels<T>(mipmapLevel: Int = 0) -> UnsafeMutablePointer<T> {
        let fromRegion = MTLRegionMake2D(0, 0, self.width, self.height)
        let bytesPerRow = 4 * self.width
        let data = UnsafeMutablePointer<T>.allocate(capacity: bytesPerRow * self.height)
        self.getBytes(data, bytesPerRow: bytesPerRow, from: fromRegion, mipmapLevel: mipmapLevel)
        return data
    }
    
    func deallocatePixels<T>(_ pixels: UnsafeMutablePointer<T>) {
        pixels.deallocate()
    }
}
