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
    private var lastPixelFormat: MTLPixelFormat?
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
        lastPixelFormat = nil
        memoryTracker.untrack(identifier: textureIdentifier)
    }
    
    private func selectOptimalPixelFormat(for device: MTLDevice) -> MTLPixelFormat {
        // Prefer compressed formats if available
        if device.supportsFamily(.apple7) {
            return .rgba8Unorm_srgb // Most efficient for modern Apple GPUs
        } else if device.supportsFamily(.apple6) {
            return .astc_4x4_srgb // Good compression, widely supported
        }
        return .rgba8Unorm // Fallback
    }
    
    func call(viewportSize: simd_float2, pixelFormat: MTLPixelFormat, commandQueue: MTLCommandQueue, uniforms: NoiseUniforms) -> MTLTexture? {
        guard uniforms.noiseAlpha > 0 else { 
            purgeTextures()
            return nil 
        }
        
        // Check if we can reuse existing texture
        if lastViewportSize == viewportSize && 
           lastPixelFormat == pixelFormat,
           let existingTexture = _noiseTexture {
            return existingTexture
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)
        
        // Clear old texture and its memory tracking
        purgeTextures()
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: selectOptimalPixelFormat(for: device),
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget, .pixelFormatView]
        // Use compressed storage where possible
        textureDescriptor.storageMode = device.hasUnifiedMemory ? .shared : .private
        
        guard let noiseTexture = device.makeTexture(descriptor: textureDescriptor),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }
        
        // Track memory usage - estimate based on format and dimensions
        let bytesPerPixel = noiseTexture.pixelFormat == .rgba8Unorm ? 4 : 2 // Compressed formats use ~2 bytes/pixel
        let memoryUsage = width * height * bytesPerPixel
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
        lastPixelFormat = pixelFormat
        _noiseTexture = noiseTexture
        
        #if DEBUG
        print("Created new noise texture - Metal Memory Usage Report:\n\(memoryTracker.memoryReport())")
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
