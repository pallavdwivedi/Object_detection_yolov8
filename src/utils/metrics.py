"""
Performance metrics tracking for latency, FPS, and queue statistics.
"""

import time
from collections import deque
from typing import Optional, Dict
from threading import Lock


class MetricsTracker:
    """
    Thread-safe metrics tracker for inference performance monitoring.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self.lock = Lock()
        
        # Latency tracking
        self.latencies = deque(maxlen=window_size)
        
        # FPS tracking
        self.frame_timestamps = deque(maxlen=window_size)
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Frame counters
        self.total_frames = 0
        self.dropped_frames = 0
        
        # Queue depth tracking
        self.queue_depths = deque(maxlen=window_size)
        
        # Start time
        self.start_time = time.time()
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement in milliseconds."""
        with self.lock:
            self.latencies.append(latency_ms)
    
    def record_frame(self):
        """Record a processed frame timestamp for FPS calculation."""
        with self.lock:
            current_time = time.time()
            self.frame_timestamps.append(current_time)
            self.total_frames += 1
            
            # Update FPS every 0.5 seconds
            if current_time - self.last_fps_update >= 0.5:
                self._update_fps()
                self.last_fps_update = current_time
    
    def record_dropped_frame(self):
        """Record a dropped frame."""
        with self.lock:
            self.dropped_frames += 1
    
    def record_queue_depth(self, depth: int):
        """Record current queue depth."""
        with self.lock:
            self.queue_depths.append(depth)
    
    def _update_fps(self):
        """Internal method to calculate FPS from timestamps."""
        if len(self.frame_timestamps) < 2:
            self.current_fps = 0.0
            return
        
        # Calculate FPS from time difference between first and last frame
        time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if time_span > 0:
            self.current_fps = len(self.frame_timestamps) / time_span
        else:
            self.current_fps = 0.0
    
    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        with self.lock:
            if not self.latencies:
                return 0.0
            return sum(self.latencies) / len(self.latencies)
    
    def get_min_latency(self) -> float:
        """Get minimum latency in milliseconds."""
        with self.lock:
            if not self.latencies:
                return 0.0
            return min(self.latencies)
    
    def get_max_latency(self) -> float:
        """Get maximum latency in milliseconds."""
        with self.lock:
            if not self.latencies:
                return 0.0
            return max(self.latencies)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        with self.lock:
            return self.current_fps
    
    def get_avg_queue_depth(self) -> float:
        """Get average queue depth."""
        with self.lock:
            if not self.queue_depths:
                return 0.0
            return sum(self.queue_depths) / len(self.queue_depths)
    
    def get_drop_rate(self) -> float:
        """Get frame drop rate as percentage."""
        with self.lock:
            total = self.total_frames + self.dropped_frames
            if total == 0:
                return 0.0
            return (self.dropped_frames / total) * 100
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        with self.lock:
            return {
                "uptime_seconds": round(self.get_uptime(), 1),
                "total_frames": self.total_frames,
                "dropped_frames": self.dropped_frames,
                "drop_rate_percent": round(self.get_drop_rate(), 2),
                "current_fps": round(self.current_fps, 1),
                "avg_latency_ms": round(self.get_avg_latency(), 1),
                "min_latency_ms": round(self.get_min_latency(), 1),
                "max_latency_ms": round(self.get_max_latency(), 1),
                "avg_queue_depth": round(self.get_avg_queue_depth(), 1)
            }
    
    def print_summary(self):
        """Print formatted metrics summary."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*50)
        print(f"Uptime:          {summary['uptime_seconds']:.1f}s")
        print(f"Total Frames:    {summary['total_frames']}")
        print(f"Dropped Frames:  {summary['dropped_frames']} ({summary['drop_rate_percent']:.2f}%)")
        print(f"Current FPS:     {summary['current_fps']:.1f}")
        print(f"Avg Latency:     {summary['avg_latency_ms']:.1f} ms")
        print(f"Min Latency:     {summary['min_latency_ms']:.1f} ms")
        print(f"Max Latency:     {summary['max_latency_ms']:.1f} ms")
        print(f"Avg Queue Depth: {summary['avg_queue_depth']:.1f}")
        print("="*50 + "\n")
