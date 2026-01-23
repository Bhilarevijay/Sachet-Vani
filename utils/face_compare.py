"""
Face Comparison Utility for Sachet

This module provides optional face comparison functionality using the face_recognition library.
If face_recognition is not installed, all functions will gracefully return None.

Installation (optional - requires dlib and cmake):
    pip install face-recognition

Note: face_recognition requires:
    - CMake (system dependency)
    - C++ compiler
    - dlib library (~500MB+ install)
"""

import os
import logging

logger = logging.getLogger(__name__)

# Try to import face_recognition - make it optional
FACE_RECOGNITION_AVAILABLE = False
face_recognition = None

try:
    import face_recognition as fr
    face_recognition = fr
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("✅ Face recognition library loaded successfully")
except ImportError:
    logger.warning("⚠️ face_recognition not installed - face comparison disabled")
except Exception as e:
    logger.warning(f"⚠️ face_recognition failed to load: {e}")


def is_available():
    """Check if face recognition is available"""
    return FACE_RECOGNITION_AVAILABLE


def compare_faces(reference_image_path, comparison_image_path, tolerance=0.6):
    """
    Compare two face images and return a similarity score.
    
    Args:
        reference_image_path: Path to the reference image (e.g., missing child photo)
        comparison_image_path: Path to comparison image (e.g., sighting photo)
        tolerance: How strict the comparison is (lower = stricter). Default 0.6
        
    Returns:
        float: Match score (0-100) or None if comparison failed/unavailable
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    
    try:
        # Handle URLs (Cloudinary) - download first
        if reference_image_path.startswith('http'):
            reference_image_path = _download_temp_image(reference_image_path)
            if not reference_image_path:
                return None
        
        if comparison_image_path.startswith('http'):
            comparison_image_path = _download_temp_image(comparison_image_path)
            if not comparison_image_path:
                return None
        
        # Check files exist
        if not os.path.exists(reference_image_path):
            logger.warning(f"Reference image not found: {reference_image_path}")
            return None
        
        if not os.path.exists(comparison_image_path):
            logger.warning(f"Comparison image not found: {comparison_image_path}")
            return None
        
        # Load images
        reference_image = face_recognition.load_image_file(reference_image_path)
        comparison_image = face_recognition.load_image_file(comparison_image_path)
        
        # Get face encodings
        reference_encodings = face_recognition.face_encodings(reference_image)
        comparison_encodings = face_recognition.face_encodings(comparison_image)
        
        # Check if faces were found
        if not reference_encodings:
            logger.warning("No face detected in reference image")
            return None
        
        if not comparison_encodings:
            logger.warning("No face detected in comparison image")
            return None
        
        # Compare faces (use first face found in each image)
        reference_encoding = reference_encodings[0]
        comparison_encoding = comparison_encodings[0]
        
        # Calculate face distance (lower = more similar)
        face_distance = face_recognition.face_distance([reference_encoding], comparison_encoding)[0]
        
        # Convert distance to a 0-100 match score
        # face_distance of 0 = perfect match (100%), distance of 1 = no match (0%)
        # Using a sigmoid-like transformation for better score distribution
        if face_distance >= 1.0:
            match_score = 0.0
        else:
            # Convert distance to percentage (1 - distance) * 100
            match_score = max(0, min(100, (1 - face_distance) * 100))
        
        logger.info(f"Face comparison: distance={face_distance:.3f}, score={match_score:.1f}%")
        
        return round(match_score, 1)
        
    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        return None


def _download_temp_image(url):
    """Download image from URL to temp file"""
    try:
        import requests
        import tempfile
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Determine extension from content type
        content_type = response.headers.get('content-type', '')
        ext = '.jpg'
        if 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        
        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def detect_faces(image_path):
    """
    Detect faces in an image and return count.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        int: Number of faces detected, or None if unavailable
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    
    try:
        if image_path.startswith('http'):
            image_path = _download_temp_image(image_path)
            if not image_path:
                return None
        
        if not os.path.exists(image_path):
            return None
        
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        return len(face_locations)
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return None
