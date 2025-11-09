"""
Thread-safe bounded queue using Python's standard queue.Queue
"""

import queue
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BoundedFrameQueue:
    """
    Wrapper around queue.Queue with frame dropping.
    """
    
    def __init__(self, max_size: int = 20, drop_policy: str = "oldest"):
        self.max_size = max_size
        self.drop_policy = drop_policy
        self.queue = queue.Queue(maxsize=max_size)
        self.frames_added = 0
        self.frames_dropped = 0
        
        logger.info(f"BoundedFrameQueue initialized: max_size={max_size}, policy={drop_policy}")
    
    def put(self, item) -> bool:
        """Add item, dropping if full."""
        try:
            self.queue.put(item, block=False)
            self.frames_added += 1
            logger.debug(f"Frame added. Queue size now: {self.queue.qsize()}")
            return True
        except queue.Full:
            logger.warning("PUT: queue full, dropping item")
            # Queue full - drop frame
            if self.drop_policy == "oldest":
                # Remove oldest and add new
                try:
                    self.queue.get(block=False)
                    self.queue.put(item, block=False)
                    self.frames_dropped += 1
                    return True
                except:
                    self.frames_dropped += 1
                    return False
            else:
                # Drop newest
                self.frames_dropped += 1
                return False
    
    def get(self, timeout: Optional[float] = None):
        """Get item from queue."""
        try:
            return self.queue.get(block=True, timeout=timeout)
            logger.debug(f"Frame removed. Queue size now: {self.queue.qsize()}")
        except queue.Empty:
            logger.debug("GET: queue empty")
            return None
    
    def size(self) -> int:
        """Get current size."""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if empty."""
        return self.queue.empty()
    
    def clear(self):
        """Clear queue."""
        while not self.queue.empty():
            try:
                self.queue.get(block=False)
            except queue.Empty:
                break
        logger.info("Queue cleared")
