"""
Main client entry point.
Orchestrates frame capture, sending, and result handling.
"""

import signal
import sys
import threading
import time
from queue import Queue
from pathlib import Path

from src.utils.logger import setup_logger
from src.client.capture_worker import CaptureWorker
from src.client.result_handler import ResultHandler
from src.communication.zmq_client import ZMQClient
from src.client.visualizer import Visualizer  # ← NEW: Import added

# Initialize logger
logger = setup_logger(__name__, log_level="DEBUG", log_file="client.log")


class InferenceClient:
    """
    Real-time inference client with capture and result handling.
    """
    
    def __init__(self, config: dict):
        """
        Initialize inference client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stop_event = threading.Event()
        
        # Queues
        self.frame_queue = Queue(maxsize=config.get('queue_size', 10))
        self.result_queue = Queue(maxsize=config.get('queue_size', 10))
        
        # ZMQ client
        self.zmq_client = ZMQClient(
            server_host=config.get('server_host', 'localhost'),
            send_port=config.get('send_port', 5555),
            recv_port=config.get('recv_port', 5556)
        )
        
        # Capture worker
        self.capture_worker = CaptureWorker(
            stream_url=config.get('stream_url', '0'),
            stream_name=config.get('stream_name', 'cam_1'),
            frame_queue=self.frame_queue,
            target_fps=config.get('target_fps', 30),
            reconnect_interval=config.get('reconnect_interval', 2),
            stop_event=self.stop_event
        )
        
        # Result handler
        self.result_handler = ResultHandler(
            result_queue=self.result_queue,
            output_dir=config.get('output_dir', 'output'),
            stop_event=self.stop_event
        )
        
        # ← NEW: Visualizer (added after result_handler)
        # #self.visualizer = Visualizer(
        #     frame_queue=self.frame_queue,
        #     result_queue=self.result_queue,
        #     stop_event=self.stop_event
        # )
        
        logger.info("InferenceClient initialized")
    
    def start(self):
        """Start the inference client."""
        try:
            logger.info("="*60)
            logger.info("STARTING INFERENCE CLIENT")
            logger.info("="*60)
            
            # Connect to server
            logger.info("Connecting to server...")
            if not self.zmq_client.connect():
                logger.error("Failed to connect to server")
                return
            
            # Start capture worker
            logger.info("Starting capture worker...")
            self.capture_worker.start()
            
            # Start result handler
            logger.info("Starting result handler...")
            self.result_handler.start()
            
            # ← NEW: Start visualizer (added after result_handler.start())
            #logger.info("Starting visualizer...")
            #self.visualizer.start()
            
            logger.info("="*60)
            logger.info("CLIENT READY - Streaming frames...")
            logger.info("="*60)
            
            # Main loop: send frames and receive results
            frames_sent = 0
            results_received = 0
            
            while not self.stop_event.is_set():
                try:
                    # Send frames
                    if not self.frame_queue.empty():
                        frame_data = self.frame_queue.get(timeout=0.01)
                        
                        # Encode frame to reduce size
                        from src.utils.helpers import encode_frame
                        try:
                            frame_data['frame'] = encode_frame(frame_data['frame'], quality=80)
                            frame_data['encoded'] = True
                        except:
                            logger.warning("Failed to encode frame, skipping")
                            continue
                        
                        if self.zmq_client.send_frame(frame_data):
                            frames_sent += 1

                            
                            if frames_sent % 100 == 0:
                                logger.info(f"Frames sent: {frames_sent}")
                    
                    # Receive results
                    result_data = self.zmq_client.receive_result(timeout=10)
                    if result_data:
                        self.result_queue.put(result_data)
                        results_received += 1
                        
                        if results_received % 100 == 0:
                            logger.info(f"Results received: {results_received}")
                    
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Client loop error: {e}", exc_info=True)
                    continue
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"Client error: {e}", exc_info=True)
            self.stop()
    
    def stop(self):
        """Stop the client gracefully."""
        logger.info("Shutting down client...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers
        self.capture_worker.join(timeout=2.0)
        self.result_handler.join(timeout=2.0)
        #self.visualizer.join(timeout=2.0)  # ← NEW: Wait for visualizer
        
        # Disconnect ZMQ
        self.zmq_client.disconnect()
        
        logger.info("Client shutdown complete")
        sys.exit(0)


def main():
    """Main entry point."""
    # Simple config (later load from YAML)
    config = {
        'stream_url': '0',  # Webcam (use RTSP URL for IP camera)
        'stream_name': 'cam_1',
        'server_host': 'localhost',
        'send_port': 5555,
        'recv_port': 5556,
        'target_fps': 30,
        'reconnect_interval': 2,
        'queue_size': 10,
        'output_dir': 'output'
    }
    
    # Create client
    client = InferenceClient(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Interrupt received, stopping...")
        client.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start client
    client.start()


if __name__ == "__main__":
    main()
