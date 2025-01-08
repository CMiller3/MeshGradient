import Metal

final class MTLBufferPool {
    private struct BufferKey: Hashable {
        let length: Int
        let options: MTLResourceOptions
    }
    
    private struct BufferEntry {
        let buffer: MTLBuffer
        let lastUsed: Date
    }
    
    private var pool: [BufferKey: [BufferEntry]] = [:]
    private let device: MTLDevice
    private let monitor = NSObject()
    private let maxBuffersPerSize = 3  // Reduced from 5
    private let maxBufferAge: TimeInterval = 30  // Seconds before buffer is eligible for cleanup
    private var memoryTracker = MetalMemoryTracker.shared
    private let poolIdentifier = "BufferPool"
    
    init(device: MTLDevice) {
        self.device = device
        
        // Start automatic cleanup timer
        Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            self?.cleanupOldBuffers()
        }
    }
    
    private func updateMemoryTracking() {
        let totalBytes = pool.values.flatMap { $0 }.reduce(0) { sum, entry in
            sum + entry.buffer.length
        }
        memoryTracker.track(bytes: totalBytes, identifier: poolIdentifier)
    }
    
    subscript(length: Int, resourceOptions: MTLResourceOptions) -> MTLBuffer? {
        get {
            objc_sync_enter(monitor)
            defer { 
                objc_sync_exit(monitor)
                updateMemoryTracking()
            }
            
            let key = BufferKey(length: length, options: resourceOptions)
            var entries = pool[key, default: []]
            
            if entries.isEmpty {
                return device.makeBuffer(length: length, options: resourceOptions)
            } else {
                let entry = entries.removeLast()
                pool[key] = entries
                return entry.buffer
            }
        }
        set {
            objc_sync_enter(monitor)
            defer { 
                objc_sync_exit(monitor)
                updateMemoryTracking()
            }
            
            guard let newValue = newValue else { return }
            let key = BufferKey(length: length, options: resourceOptions)
            var entries = pool[key, default: []]
            
            if entries.count < maxBuffersPerSize {
                entries.append(BufferEntry(buffer: newValue, lastUsed: Date()))
                pool[key] = entries
            }
            // If we exceed maxBuffersPerSize, let the buffer be deallocated
        }
    }
    
    private func cleanupOldBuffers() {
        objc_sync_enter(monitor)
        defer { 
            objc_sync_exit(monitor)
            updateMemoryTracking()
        }
        
        let now = Date()
        for (key, entries) in pool {
            let validEntries = entries.filter { now.timeIntervalSince($0.lastUsed) < maxBufferAge }
            if validEntries.count != entries.count {
                pool[key] = validEntries
            }
        }
        
        #if DEBUG
        print("Buffer Pool Memory Usage: \(Double(memoryTracker.totalMemoryUsage()) / 1_000_000.0)MB")
        #endif
    }
    
    func cleanup() {
        objc_sync_enter(monitor)
        defer { 
            objc_sync_exit(monitor)
            memoryTracker.untrack(identifier: poolIdentifier)
        }
        pool.removeAll()
    }
    
    deinit {
        cleanup()
    }
}
