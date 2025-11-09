"""
Helper utilities for frame preprocessing and common operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Preprocess frame for YOLO inference.
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        target_size: Target (width, height) for resizing
        maintain_aspect: If True, maintain aspect ratio with padding
    
    Returns:
        Preprocessed frame ready for inference
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame: empty or None")
    
    # Convert BGR to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if maintain_aspect:
        # Resize maintaining aspect ratio
        frame_resized = letterbox_resize(frame_rgb, target_size)
    else:
        # Simple resize (may distort image)
        frame_resized = cv2.resize(frame_rgb, target_size)
    
    return frame_resized


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding (letterboxing).
    
    Args:
        image: Input image (RGB format)
        target_size: Target (width, height)
        color: Padding color (RGB)
    
    Returns:
        Resized and padded image
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Calculate scaling ratio
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas with padding color
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return canvas


def encode_frame(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    Encode frame to JPEG bytes for network transmission.
    
    Args:
        frame: Input frame (numpy array)
        quality: JPEG quality (1-100, higher = better quality but larger size)
    
    Returns:
        Encoded frame as bytes
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', frame, encode_param)
    
    if not success:
        raise RuntimeError("Failed to encode frame")
    
    return encoded.tobytes()


def decode_frame(frame_bytes: bytes) -> np.ndarray:
    """
    Decode JPEG bytes back to numpy array.
    
    Args:
        frame_bytes: Encoded frame bytes
    
    Returns:
        Decoded frame as numpy array
    """
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise RuntimeError("Failed to decode frame")
    
    return frame


def format_detection_output(
    results,
    stream_name: str,
    frame_id: int,
    timestamp: float,
    latency_ms: float
) -> dict:
    """
    Format YOLO detection results into required JSON structure.
    
    Args:
        results: YOLO results object
        stream_name: Camera stream identifier
        frame_id: Frame sequence number
        timestamp: Unix timestamp
        latency_ms: End-to-end latency in milliseconds
    
    Returns:
        Formatted detection dictionary
    """
    detections = []
    
    # Extract detections from YOLO results
    if results and len(results) > 0:
        result = results[0]  # YOLO returns list, take first result
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                label = result.names[cls]
                
                detections.append({
                    "label": label,
                    "conf": round(conf, 2),
                    "bbox": [float(x) for x in box]  # [x1, y1, x2, y2]
                })
    
    return {
        "timestamp": int(timestamp),
        "frame_id": frame_id,
        "stream_name": stream_name,
        "latency_ms": round(latency_ms, 1),
        "detections": detections
    }
