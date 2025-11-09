"""
Real-time visualization of detection results.
"""
import cv2
import json
import threading
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer(threading.Thread):
    """
    Display frames with bounding boxes in real-time.
    """
    
    def __init__(self, frame_queue, result_queue, stop_event):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.latest_result = None
        
    def run(self):
        logger.info("Visualizer started")
        
        while not self.stop_event.is_set():
            try:
                # Get latest frame
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=0.01)
                    frame = frame_data['frame'].copy()
                    frame_id = frame_data['frame_id']
                    
                    # Draw bounding boxes if we have results
                    if self.latest_result and self.latest_result['frame_id'] == frame_id:
                        frame = self._draw_detections(frame, self.latest_result['result'])
                    
                    # Show frame
                    cv2.imshow('Real-Time Inference', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit signal received")
                        self.stop_event.set()
                        break
                
                # Update latest result
                if not self.result_queue.empty():
                    self.latest_result = self.result_queue.get(timeout=0.01)
                    
            except Exception as e:
                logger.error(f"Visualizer error: {e}")
                continue
        
        cv2.destroyAllWindows()
        logger.info("Visualizer stopped")
    
    def _draw_detections(self, frame, result):
        """Draw bounding boxes on frame."""
        for detection in result['detections']:
            bbox = detection['bbox']
            label = detection['label']
            conf = detection['conf']
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
