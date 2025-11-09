"""
Camera capture worker for reading frames from webcam or RTSP streams.
Handles reconnection and frame rate control.
"""

import cv2
import time
import threading
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaptureWorker(threading.Thread):
    """
    Worker thread that captures frames from video source.
    """
    
    def __init__(
        self,
        stream_url: str,
        stream_name: str,
        frame_queue,
        target_fps: int = 30,
        reconnect_interval: int = 2,
        stop_event: threading.Event = None
    ):
        """
        Initialize capture worker.
        
        Args:
            stream_url: Video source (0 for webcam, RTSP URL for IP camera)
            stream_name: Unique stream identifier
            frame_queue: Queue to push captured frames to
            target_fps: Target frame rate for capture
            reconnect_interval: Seconds to wait before reconnection attempt
            stop_event: Event to signal worker shutdown
        """
        super().__init__(daemon=True)
        self.stream_url = stream_url
        self.stream_name = stream_name
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        self.reconnect_interval = reconnect_interval
        self.stop_event = stop_event or threading.Event()
        
        self.frame_id = 0
        self.capture = None
        self.reconnect_attempts = 0
        
        logger.info(f"CaptureWorker initialized: {stream_name} -> {stream_url}")
    
    def _connect(self) -> bool:
        """
        Connect to video source.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"{self.stream_name}: Connecting to {self.stream_url}...")
            
            # Convert string "0" to integer for webcam
            if self.stream_url == "0" or self.stream_url == 0:
                self.capture = cv2.VideoCapture(0)
            else:
                self.capture = cv2.VideoCapture(self.stream_url)
            
            # Set buffer size to 1 to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.capture.isOpened():
                logger.error(f"{self.stream_name}: Failed to open video source")
                return False
            
            # Test read
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"{self.stream_name}: Failed to read test frame")
                self.capture.release()
                return False
            if self.frame_id % 50 == 0:  # Every 50 frames
                logger.info(f"Captured frame {self.frame_id}: shape={frame.shape}, size={frame.nbytes} bytes")

            logger.info(f"{self.stream_name}: Connected successfully (resolution: {frame.shape[1]}x{frame.shape[0]})")
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            logger.error(f"{self.stream_name}: Connection error: {e}")
            return False
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        
        Returns:
            True if reconnection successful
        """
        if self.capture:
            self.capture.release()
        
        self.reconnect_attempts += 1
        wait_time = min(self.reconnect_interval * (2 ** (self.reconnect_attempts - 1)), 30)
        
        logger.warning(
            f"{self.stream_name}: Reconnection attempt #{self.reconnect_attempts} "
            f"in {wait_time}s..."
        )
        
        time.sleep(wait_time)
        return self._connect()
    
    def run(self):
        """
        Main capture loop.
        """
        logger.info(f"{self.stream_name}: Capture worker started")
        
        # Initial connection
        if not self._connect():
            logger.error(f"{self.stream_name}: Failed to establish initial connection")
            return
        
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue
                
                # Capture frame
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    logger.warning(f"{self.stream_name}: Failed to read frame, reconnecting...")
                    if not self._reconnect():
                        logger.error(f"{self.stream_name}: Reconnection failed, retrying...")
                        continue
                    continue
                
                # Resize frame to 640x640 to reduce data size
                frame_resized = cv2.resize(frame, (640, 640))


                # SAVE FIRST 5 FRAMES FOR DEBUG
                if self.frame_id < 5:
                    import os
                    os.makedirs("debug_frames", exist_ok=True)
                    cv2.imwrite(f"debug_frames/frame_{self.frame_id}.jpg", frame_resized)
                    logger.info(f"Saved debug frame: debug_frames/frame_{self.frame_id}.jpg")

                # Prepare frame data
                frame_data = {
                    'frame': frame_resized,
                    'stream_name': self.stream_name,
                    'frame_id': self.frame_id,
                    'timestamp': time.time()
                }

                # Push to queue
                self.frame_queue.put(frame_data)

                # # Prepare frame data
                # frame_data = {
                #     'frame': frame,
                #     'stream_name': self.stream_name,
                #     'frame_id': self.frame_id,
                #     'timestamp': time.time()
                # }
                
                # # Push to queue
                # self.frame_queue.put(frame_data)
                
                self.frame_id += 1
                last_frame_time = current_time
                
                logger.debug(f"{self.stream_name}: Captured frame #{self.frame_id}")
                
            except Exception as e:
                logger.error(f"{self.stream_name}: Capture error: {e}", exc_info=True)
                time.sleep(1)
                continue
        
        # Cleanup
        if self.capture:
            self.capture.release()
        
        logger.info(f"{self.stream_name}: Capture worker stopped")
    
    def stop(self):
        """Stop the capture worker."""
        logger.info(f"{self.stream_name}: Stopping capture worker...")
        self.stop_event.set()
