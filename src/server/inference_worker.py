"""
Inference worker threads that process frames from the queue.
Multiple workers run in parallel to maximize CPU utilization.
"""
import os 
import time
import threading
import numpy as np
from typing import Optional
from src.utils.logger import get_logger
from src.utils.metrics import MetricsTracker
from src.utils.helpers import format_detection_output, decode_frame
from src.server.frame_queue import BoundedFrameQueue

logger = get_logger(__name__)


class InferenceWorker(threading.Thread):
    """
    Worker thread that pulls frames from queue and runs inference.
    """
    
    def __init__(
        self,
        worker_id: int,
        model,
        input_queue: BoundedFrameQueue,
        output_queue: BoundedFrameQueue,
        metrics: MetricsTracker,
        stop_event: threading.Event
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.metrics = metrics
        self.stop_event = stop_event
        
        logger.info(f"InferenceWorker-{worker_id} initialized")

    import os
    def run(self):
        """
        Main worker loop: pull frame -> inference -> push result.
        """
        logger.info(f"[Worker-{self.worker_id}] PID={os.getpid()} id(input_queue)={id(self.input_queue)}")
        print(f"[Worker-{self.worker_id}] input_queue id: {id(self.input_queue)}", flush=True)
        
        logger.info(f"InferenceWorker-{self.worker_id} started")

        logger.info(f"Worker-{self.worker_id}: stop_event.is_set() = {self.stop_event.is_set()}")
        logger.info(f"Worker-{self.worker_id}: input_queue size = {self.input_queue.size()}")
    
        
        frame_count = 0
        
        while not self.stop_event.is_set():
           
            
            try:
                # Get frame from queue
                print(f"Worker-{self.worker_id}: queue size before get: {self.input_queue.size()}")
                frame_data = self.input_queue.get(timeout=0.1)
                print(f"Worker-{self.worker_id}: get() returned {frame_data is not None}")  # ADD THIS
                
                if frame_data is None:
                    continue
                
                # Unpack frame data
                frame = frame_data.get('frame')
                stream_name = frame_data.get('stream_name', 'unknown')
                frame_id = frame_data.get('frame_id', 0)
                capture_time = frame_data.get('timestamp', time.time())
                is_encoded = frame_data.get('encoded', False)
                
                logger.info(f"Worker-{self.worker_id}: GOT FRAME {frame_id}, encoded={is_encoded}, type={type(frame)}")
                
                if frame is None:
                    logger.warning(f"Worker-{self.worker_id}: Frame is None, skipping")
                    continue
                
                # Decode if needed
                if is_encoded and isinstance(frame, bytes):
                    try:
                        frame = decode_frame(frame)
                        logger.info(f"Worker-{self.worker_id}: Decoded frame {frame_id} to shape {frame.shape}")
                    except Exception as e:
                        logger.error(f"Worker-{self.worker_id}: Decode failed: {e}")
                        continue
                
                # Verify frame is valid numpy array
                if not isinstance(frame, np.ndarray):
                    logger.error(f"Worker-{self.worker_id}: Frame is not numpy array: {type(frame)}")
                    continue
                
                logger.info(f"Worker-{self.worker_id}: Running inference on frame {frame_id}, shape={frame.shape}")
                
                # Record queue depth
                self.metrics.record_queue_depth(self.input_queue.size())
                
                # Run inference
                inference_start = time.time()
                results = self.model(frame, verbose=False)
                inference_end = time.time()
                
                inference_latency = (inference_end - inference_start) * 1000  # ms
                e2e_latency = (inference_end - capture_time) * 1000  # ms
                
                logger.info(f"Worker-{self.worker_id}: Inference done for frame {frame_id} in {inference_latency:.1f}ms")
                
                # Format results
                result_data = format_detection_output(
                    results=results,
                    stream_name=stream_name,
                    frame_id=frame_id,
                    timestamp=capture_time,
                    latency_ms=e2e_latency
                )
                
                # Add to output queue
                output_data = {
                    'result': result_data,
                    'stream_name': stream_name,
                    'frame_id': frame_id
                }
                self.output_queue.put(output_data)
                
                # Update metrics
                self.metrics.record_latency(e2e_latency)
                self.metrics.record_frame()
                
                frame_count += 1
                
                logger.info(f"Worker-{self.worker_id}: Completed frame {frame_id} (total: {frame_count})")
                
            except Exception as e:
                logger.error(f"Worker-{self.worker_id} ERROR: {e}", exc_info=True)
                continue
        
        logger.info(f"InferenceWorker-{self.worker_id} stopped (processed {frame_count} frames)")
