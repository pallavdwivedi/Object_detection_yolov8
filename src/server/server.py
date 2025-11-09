"""
Main server entry point.
Orchestrates model loading, worker threads, and ZeroMQ communication.
"""

import signal
import sys
import threading
import time
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker
from src.server.model_loader import YOLOModelLoader
from src.server.frame_queue import BoundedFrameQueue
from src.server.inference_worker import InferenceWorker
from src.communication.zmq_server import ZMQServer

# Initialize logger
logger = setup_logger(__name__, log_level="DEBUG", log_file="server.log")


class InferenceServer:
    """
    Real-time inference server with multi-threaded processing.
    """
    
    def __init__(self, config: dict):
        """
        Initialize inference server.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stop_event = threading.Event()
        self.workers = []
        self.metrics = MetricsTracker()
        
        # Queues
        self.input_queue = BoundedFrameQueue(
            max_size=config.get('max_queue_size', 20),
            drop_policy=config.get('drop_policy', 'oldest')
        )
        self.output_queue = BoundedFrameQueue(
            max_size=config.get('max_queue_size', 20),
            drop_policy='oldest'
        )
        
        # Model loader
        self.model_loader = YOLOModelLoader(
            model_path=config.get('model_path', 'yolov8n.pt'),
            device=config.get('device', 'cpu'),
            img_size=config.get('img_size', 640),
            conf_threshold=config.get('conf_threshold', 0.25),
            iou_threshold=config.get('iou_threshold', 0.45)
        )
        
        # ZMQ server
        self.zmq_server = ZMQServer(
            recv_port=config.get('recv_port', 5555),
            send_port=config.get('send_port', 5556),
            input_queue=self.input_queue,
            output_queue=self.output_queue
        )
        
        logger.info("InferenceServer initialized")
    
    def start(self):
        """Start the inference server."""
        try:
            logger.info("="*60)
            logger.info("STARTING REAL-TIME INFERENCE SERVER")
            logger.info("="*60)
            
            # Load model
            logger.info("Loading YOLO model...")
            model = self.model_loader.load_model()
            logger.info(f"Model info: {self.model_loader.get_model_info()}")
            
            # Start ZMQ server
            logger.info("Starting ZMQ communication...")
            self.zmq_server.start()
            
            # Spawn inference workers
            num_workers = self.config.get('num_workers', 4)
            logger.info(f"Spawning {num_workers} inference workers...")
            
            for i in range(num_workers):
                worker = InferenceWorker(
                    worker_id=i,
                    model=model,
                    input_queue=self.input_queue,
                    output_queue=self.output_queue,
                    metrics=self.metrics,
                    stop_event=self.stop_event
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"All {num_workers} workers started")

            import time
            time.sleep(2)  # Give workers time to start
            for worker in self.workers:
                logger.info(f"Worker {worker.worker_id}: is_alive={worker.is_alive()}, daemon={worker.daemon}")
                        
            # Start result sender in separate thread
            result_sender_thread = threading.Thread(
                target=self.zmq_server.send_results,
                daemon=True
            )
            result_sender_thread.start()
            
            # Start metrics printer
            metrics_thread = threading.Thread(
                target=self._print_metrics_loop,
                daemon=True
            )
            metrics_thread.start()
            
            logger.info("="*60)
            logger.info("SERVER READY - Waiting for frames...")
            logger.info("="*60)
            
            # Start receiving frames (blocking)
            self.zmq_server.receive_frames()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            self.stop()
    
    def _print_metrics_loop(self):
        """Print metrics periodically."""
        while not self.stop_event.is_set():
            time.sleep(5)  # Print every 5 seconds
            self.metrics.print_summary()
    
    def stop(self):
        """Stop the server gracefully."""
        logger.info("Shutting down server...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        # Stop ZMQ server
        self.zmq_server.stop()
        
        # Final metrics
        logger.info("Final metrics:")
        self.metrics.print_summary()
        
        logger.info("Server shutdown complete")
        sys.exit(0)


def main():
    """Main entry point."""
    # Simple config (later load from YAML)
    config = {
        'model_path': 'yolov8n.pt',
        'device': 'cpu',
        'img_size': 640,
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_queue_size': 20,
        'drop_policy': 'oldest',
        'num_workers': 4,  # Adjust based on CPU cores
        'recv_port': 5555,
        'send_port': 5556
    }
    
    # Create server
    server = InferenceServer(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Interrupt received, stopping...")
        server.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    server.start()


if __name__ == "__main__":
    main()
