"""
Result handler for receiving inference results and writing JSON files.
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional
from queue import Empty
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResultHandler(threading.Thread):
    """
    Worker thread that receives results and saves them as JSON.
    """
    
    def __init__(
        self,
        result_queue,
        output_dir: str = "output",
        stop_event: threading.Event = None
    ):
        """
        Initialize result handler.
        
        Args:
            result_queue: Queue to pull results from
            output_dir: Base directory for output JSON files
            stop_event: Event to signal worker shutdown
        """
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.output_dir = Path(output_dir)
        self.stop_event = stop_event or threading.Event()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_saved = 0
        
        logger.info(f"ResultHandler initialized: output_dir={output_dir}")
    
    def run(self):
        """
        Main result handling loop.
        """
        logger.info("ResultHandler started")
        
        while not self.stop_event.is_set():
            try:
                # Get result from queue (timeout to check stop_event periodically)
                result_data = self.result_queue.get(timeout=0.1)
                
                if result_data is None:
                    continue
                
                # Extract result
                result = result_data['result']
                stream_name = result_data['stream_name']
                frame_id = result_data['frame_id']
                
                # Save to JSON
                self._save_json(result, stream_name, frame_id)
                
                self.results_saved += 1
                
                logger.debug(
                    f"Saved result: {stream_name} frame {frame_id} "
                    f"({len(result['detections'])} detections)"
                )
                
            except Empty:
                # Queue timeout - this is normal, just continue
                continue
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Result handling error: {e}", exc_info=True)
                continue
        
        logger.info(f"ResultHandler stopped (total results saved: {self.results_saved})")
    
    def _save_json(self, result: dict, stream_name: str, frame_id: int):
        """
        Save result as JSON file.
        
        Args:
            result: Detection result dictionary
            stream_name: Stream identifier
            frame_id: Frame number
        """
        try:
            # Create stream-specific subdirectory
            stream_dir = self.output_dir / stream_name
            stream_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = f"frame_{frame_id:06d}.json"
            filepath = stream_dir / filename
            
            # Write JSON
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.debug(f"Saved JSON: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON for {stream_name} frame {frame_id}: {e}")
    
    def stop(self):
        """Stop the result handler."""
        logger.info("Stopping result handler...")
        self.stop_event.set()
