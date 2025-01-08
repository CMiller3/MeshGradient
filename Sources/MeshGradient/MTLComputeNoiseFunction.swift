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
    private var _noiseTexture: MTLTexture?
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
    
    private func createTextureDescriptor(width: Int, height: Int) -> MTLTextureDescriptor {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: device.supportsFamily(.apple7) ? .rgba8Unorm_srgb : .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        descriptor.storageMode = device.hasUnifiedMemory ? .shared : .private
        return descriptor
    }
    
    func call(viewportSize: simd_float2, pixelFormat: MTLPixelFormat, commandQueue: MTLCommandQueue, uniforms: NoiseUniforms) -> MTLTexture? {
        guard uniforms.noiseAlpha > 0 else { 
            purgeTextures()
            return nil 
        }
        
        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)
        
        // Reuse existing texture if size hasn't changed
        if lastViewportSize == viewportSize, let existingTexture = _noiseTexture {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder()
            else { return existingTexture }
            
            encoder.setComputePipelineState(pipelineState)
            encoder.setTexture(existingTexture, index: Int(ComputeNoiseInputIndexOutputTexture.rawValue))
            
            var uniformsCopy = uniforms
            encoder.setBytes(&uniformsCopy,
                           length: MemoryLayout.size(ofValue: uniformsCopy),
                           index: Int(ComputeNoiseInputIndexUniforms.rawValue))
            
            let threadgroupCounts = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroups = MTLSize(
                width: (width + threadgroupCounts.width - 1) / threadgroupCounts.width,
                height: (height + threadgroupCounts.height - 1) / threadgroupCounts.height,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupCounts)
            encoder.endEncoding()
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            return existingTexture
        }
        
        // Create new texture if needed
        let descriptor = createTextureDescriptor(width: width, height: height)
        guard let noiseTexture = device.makeTexture(descriptor: descriptor),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }
        
        // Track memory usage
        let memoryUsage = width * height * 4
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
