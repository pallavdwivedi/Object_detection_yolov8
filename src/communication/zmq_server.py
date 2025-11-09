"""
ZeroMQ server socket wrapper for receiving frames and sending results.
Uses PULL socket for receiving frames and PUSH socket for sending results.
"""
import os 
import zmq
import pickle
import time
from typing import Callable, Optional
from src.utils.logger import get_logger
from src.server.frame_queue import BoundedFrameQueue

logger = get_logger(__name__)


class ZMQServer:
    """
    ZeroMQ server for frame ingestion and result distribution.
    """
    
    def __init__(
        self,
        recv_port: int = 5555,
        send_port: int = 5556,
        input_queue: Optional[BoundedFrameQueue] = None,
        output_queue: Optional[BoundedFrameQueue] = None
    ):
        """
        Initialize ZeroMQ server.
        
        Args:
            recv_port: Port to receive frames on
            send_port: Port to send results on
            input_queue: Queue to push received frames to
            output_queue: Queue to pull results from
        """
        self.recv_port = recv_port
        self.send_port = send_port
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        self.context = zmq.Context()
        self.recv_socket = None
        self.send_socket = None
        self.running = False
        
        logger.info(f"ZMQServer initialized: recv_port={recv_port}, send_port={send_port}")
    
    def start(self):
        """Start ZeroMQ sockets."""
        try:
            # Socket to receive frames (PULL pattern)
            self.recv_socket = self.context.socket(zmq.PULL)
            self.recv_socket.bind(f"tcp://*:{self.recv_port}")
            logger.info(f"Receiving socket bound to port {self.recv_port}")
            
            # Socket to send results (PUSH pattern)
            self.send_socket = self.context.socket(zmq.PUSH)
            self.send_socket.bind(f"tcp://*:{self.send_port}")
            logger.info(f"Sending socket bound to port {self.send_port}")
            
            self.running = True
            logger.info("ZMQ server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start ZMQ server: {e}")
            raise
    
    import os
    def receive_frames(self):
        """
        Receive frames and push to input queue.
        Runs in main thread (blocking).
        """
        logger.info(f"[ZMQServer] PID={os.getpid()} id(input_queue)={id(self.input_queue)}")
        print(f"[ZMQServer] input_queue id: {id(self.input_queue)}", flush=True)
        
        logger.info("Starting frame reception loop...")
        

        
        while self.running:
            try:
                # Receive frame data (non-blocking with timeout)
                if self.recv_socket.poll(timeout=100):  # 100ms timeout
                    frame_bytes = self.recv_socket.recv()
                    
                    logger.info(f"SERVER RECEIVED: {len(frame_bytes)} bytes")
                    # Deserialize frame data
                    frame_data = pickle.loads(frame_bytes)

                    # ADD THIS DEBUG:
                    logger.info(f"Unpickled frame data keys: {frame_data.keys()}")
                    logger.info(f"Frame type: {type(frame_data.get('frame'))}, shape: {frame_data.get('frame').shape if hasattr(frame_data.get('frame'), 'shape') else 'NO SHAPE'}")

                    
                    # Add reception timestamp
                    frame_data['recv_timestamp'] = time.time()
                    
                    
                    # Push to input queue
                    if self.input_queue:
                        success = self.input_queue.put(frame_data)
                        logger.info(f"Pushed to queue: {frame_data['stream_name']} #{frame_data['frame_id']}, success={success}, queue_size={self.input_queue.size()}")
                    # # Push to input queue
                    # if self.input_queue:
                    #     success = self.input_queue.put(frame_data)
                    #     if not success:
                    #         logger.warning(f"Frame dropped: queue full")
                    
                    # logger.debug(f"Received frame: {frame_data['stream_name']} #{frame_data['frame_id']}")
                    
            except Exception as e:
                logger.error(f"Error receiving frame: {e}")
                continue
    
    def send_results(self):
        """
        Pull results from output queue and send to clients.
        Should run in separate thread.
        """
        logger.info("Starting result sending loop...")
        
        while self.running:
            try:
                # Get result from output queue
                if self.output_queue:
                    result_data = self.output_queue.get(timeout=0.1)
                    
                    if result_data is None:
                        continue
                    
                    # Serialize and send
                    result_bytes = pickle.dumps(result_data)
                    self.send_socket.send(result_bytes)
                    
                    logger.debug(
                        f"Sent result: {result_data['stream_name']} "
                        f"#{result_data['frame_id']}"
                    )
                    
            except Exception as e:
                if self.running:  # Only log if not intentional shutdown
                    logger.error(f"Error sending result: {e}")
                continue
    
    def stop(self):
        """Stop server and close sockets."""
        logger.info("Stopping ZMQ server...")
        self.running = False
        
        if self.recv_socket:
            self.recv_socket.close()
        if self.send_socket:
            self.send_socket.close()
        
        self.context.term()
        logger.info("ZMQ server stopped")
