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
        // Use the most memory-efficient format that's widely supported
        if device.hasUnifiedMemory {
            return .r8Unorm // Single channel is sufficient for noise
        }
        return .r8Unorm // Fallback to same format
    }
    
    func call(viewportSize: simd_float2, pixelFormat: MTLPixelFormat, commandQueue: MTLCommandQueue, uniforms: NoiseUniforms) -> MTLTexture? {
        guard uniforms.noiseAlpha > 0 else { 
            purgeTextures()
            return nil 
        }
        
        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)
        
        if lastViewportSize == viewportSize, let existingTexture = _noiseTexture {
            return existingTexture
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Clear old texture and its memory tracking
        purgeTextures()
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: selectOptimalPixelFormat(for: device),
            width: width,
            height: height,
            mipmapped: false
        )
        
        // Simple optimization for storage mode
        textureDescriptor.storageMode = device.hasUnifiedMemory ? .shared : .private
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let noiseTexture = device.makeTexture(descriptor: textureDescriptor),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }
        
        // Track memory usage - r8Unorm uses 1 byte per pixel
        let memoryUsage = width * height
        memoryTracker.track(bytes: memoryUsage, identifier: textureIdentifier)
        
        let threadgroupCounts = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (width + threadgroupCounts.width - 1) / threadgroupCounts.width,
            height: (height + threadgroupCounts.height - 1) / threadgroupCounts.height,
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
        print(" - Size: \(width)x\(height)")
        print(" - Estimated Memory: \(Double(memoryUsage) / 1_000_000.0)MB")
        print("Metal Memory Usage Report:\n\(memoryTracker.memoryReport())")
        #endif
        
        return noiseTexture
    }
}

extension MTLTexture {
    func getPixels<T>(mipmapLevel: Int = 0) -> UnsafeMutablePointer<T> {
        let fromRegion = MTLRegionMake2D(0, 0, self.width, self.height)
        let bytesPerRow = self.width  // Adjusted for r8Unorm format
        let data = UnsafeMutablePointer<T>.allocate(capacity: bytesPerRow * self.height)
        self.getBytes(data, bytesPerRow: bytesPerRow, from: fromRegion, mipmapLevel: mipmapLevel)
        return data
    }
    
    func deallocatePixels<T>(_ pixels: UnsafeMutablePointer<T>) {
        pixels.deallocate()
    }
}
