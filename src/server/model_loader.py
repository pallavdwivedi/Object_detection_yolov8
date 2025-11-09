"""
YOLO model loader with warmup and optimization for CPU inference.
Loads model once and exposes to all worker threads.
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YOLOModelLoader:
    """
    Handles YOLO model loading, warmup, and configuration.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to YOLO weights file
            device: Device to run inference ('cpu' or 'cuda:0')
            img_size: Input image size
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        logger.info(f"Initializing YOLOModelLoader: model={model_path}, device={device}, size={img_size}")
    
    def load_model(self) -> YOLO:
        """
        Load YOLO model with optimizations.
        
        Returns:
            Loaded YOLO model
        """
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Set device
            self.model.to(self.device)
            
            # Configure model
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            
            # CPU-specific optimizations
            if self.device == "cpu":
                logger.info("Applying CPU optimizations...")
                torch.set_num_threads(torch.get_num_threads())  # Use all available threads
                
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Warmup
            self._warmup()
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _warmup(self, num_runs: int = 3):
        """
        Warm up model with dummy inference to stabilize performance.
        
        Args:
            num_runs: Number of warmup runs
        """
        logger.info(f"Warming up model with {num_runs} dummy runs...")
        
        try:
            # Create dummy input
            dummy_input = np.random.randint(
                0, 255, 
                (self.img_size, self.img_size, 3), 
                dtype=np.uint8
            )
            
            # Run inference multiple times
            for i in range(num_runs):
                _ = self.model(dummy_input, verbose=False)
                logger.debug(f"Warmup run {i+1}/{num_runs} completed")
            
            logger.info("Model warmup completed successfully")
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            raise
    
    def get_model(self) -> YOLO:
        """
        Get loaded model instance.
        
        Returns:
            YOLO model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {}
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "img_size": self.img_size,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "model_type": self.model.model.__class__.__name__
        }
