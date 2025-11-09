"""
ZeroMQ client socket wrapper for sending frames and receiving results.
"""

import zmq
import pickle
import time
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ZMQClient:
    """
    ZeroMQ client for sending frames and receiving results.
    """
    
    def __init__(
        self,
        server_host: str = "localhost",
        send_port: int = 5555,
        recv_port: int = 5556
    ):
        """
        Initialize ZeroMQ client.
        
        Args:
            server_host: Server hostname or IP
            send_port: Port to send frames to (server's recv_port)
            recv_port: Port to receive results from (server's send_port)
        """
        self.server_host = server_host
        self.send_port = send_port
        self.recv_port = recv_port
        
        self.context = zmq.Context()
        self.send_socket = None
        self.recv_socket = None
        self.connected = False
        
        logger.info(
            f"ZMQClient initialized: server={server_host}, "
            f"send_port={send_port}, recv_port={recv_port}"
        )
    
    def connect(self) -> bool:
        """
        Connect to server.
        
        Returns:
            True if connection successful
        """
        try:
            # Socket to send frames (PUSH pattern)
            self.send_socket = self.context.socket(zmq.PUSH)
            self.send_socket.connect(f"tcp://{self.server_host}:{self.send_port}")
            logger.info(f"Send socket connected to tcp://{self.server_host}:{self.send_port}")
            
            # Socket to receive results (PULL pattern)
            self.recv_socket = self.context.socket(zmq.PULL)
            self.recv_socket.connect(f"tcp://{self.server_host}:{self.recv_port}")
            logger.info(f"Recv socket connected to tcp://{self.server_host}:{self.recv_port}")
            
            self.connected = True
            logger.info("ZMQ client connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect ZMQ client: {e}")
            self.connected = False
            return False
    
    def send_frame(self, frame_data: dict, timeout: int = 100) -> bool:
        """
        Send frame data to server.
        
        Args:
            frame_data: Frame data dictionary
            timeout: Send timeout in milliseconds
        
        Returns:
            True if sent successfully
        """
        if not self.connected:
            logger.warning("Cannot send frame: not connected")
            return False
        
        try:
            # Serialize frame data
            frame_bytes = pickle.dumps(frame_data)
            
            # ADD THIS DEBUG LOG:
            logger.debug(f"Sending frame {frame_data['frame_id']}: {len(frame_bytes)} bytes")
            
            # Send with timeout (non-blocking)
            self.send_socket.send(frame_bytes, flags=zmq.NOBLOCK)
            
            logger.debug(
                f"Sent frame: {frame_data['stream_name']} "
                f"#{frame_data['frame_id']}"
            )
            return True
            
        except zmq.Again:
            # Socket buffer full (server not keeping up)
            logger.warning("Send buffer full, frame dropped")
            return False
            
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            return False
    
    def receive_result(self, timeout: int = 100) -> Optional[dict]:
        """
        Receive result from server.
        
        Args:
            timeout: Receive timeout in milliseconds
        
        Returns:
            Result dictionary or None if timeout/error
        """
        if not self.connected:
            return None
        
        try:
            # Poll for available data
            if self.recv_socket.poll(timeout=timeout):
                result_bytes = self.recv_socket.recv()
                result_data = pickle.loads(result_bytes)
                
                logger.debug(
                    f"Received result: {result_data['stream_name']} "
                    f"#{result_data['frame_id']}"
                )
                return result_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error receiving result: {e}")
            return None
    
    def disconnect(self):
        """Disconnect and close sockets."""
        logger.info("Disconnecting ZMQ client...")
        
        if self.send_socket:
            self.send_socket.close()
        if self.recv_socket:
            self.recv_socket.close()
        
        self.context.term()
        self.connected = False
        logger.info("ZMQ client disconnected")
