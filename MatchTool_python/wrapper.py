"""
High-level Python wrapper for the Fastest Image Pattern Matching library.

This module provides a user-friendly interface for fast template matching
with SIMD optimizations.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
import sys
import os

# Add the current directory to path for importing the compiled module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .core_matcher import (
        TemplateMatcher as _TemplateMatcher,
        MatchParameters as _MatchParameters,
        MatchResult as _MatchResult,
        get_version,
        has_simd_support
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import core_matcher module. Make sure the library is compiled.\n"
        f"Error: {e}\n"
        "Try running: pip install -e ."
    )


class MatchResult:
    """Result of template matching."""
    
    def __init__(self, position: Tuple[float, float], score: float, angle: float = 0.0):
        self.position = position  # (x, y) center position
        self.score = score      # Match score (0.0 to 1.0)
        self.angle = angle      # Rotation angle in degrees
    
    def __repr__(self):
        return f"MatchResult(position={self.position}, score={self.score:.3f}, angle={self.angle:.1f})"


class MatchParameters:
    """Parameters for template matching."""
    
    def __init__(
        self,
        max_positions: int = 10,
        max_overlap: float = 0.1,
        score_threshold: float = 0.8,
        tolerance_angle: float = 5.0,
        min_reduce_area: int = 1000,
        use_simd: bool = True,
        enable_subpixel: bool = False,
        fast_mode: bool = False
    ):
        self.max_positions = max_positions
        self.max_overlap = max_overlap
        self.score_threshold = score_threshold
        self.tolerance_angle = tolerance_angle
        self.min_reduce_area = min_reduce_area
        self.use_simd = use_simd
        self.enable_subpixel = enable_subpixel
        self.fast_mode = fast_mode
    
    def _to_cpp(self) -> '_MatchParameters':
        """Convert to C++ MatchParameters."""
        cpp_params = _MatchParameters()
        cpp_params.max_positions = self.max_positions
        cpp_params.max_overlap = self.max_overlap
        cpp_params.score_threshold = self.score_threshold
        cpp_params.tolerance_angle = self.tolerance_angle
        cpp_params.min_reduce_area = self.min_reduce_area
        cpp_params.use_simd = self.use_simd
        cpp_params.enable_subpixel = self.enable_subpixel
        cpp_params.fast_mode = self.fast_mode
        return cpp_params


class TemplateMatcher:
    """
    Fast template matching with SIMD optimizations.
    
    This class provides high-performance template matching using the
    fastest available SIMD instructions (SSE2, AVX2, or NEON).
    """
    
    def __init__(self, parameters: Optional[MatchParameters] = None):
        """
        Initialize the template matcher.
        
        Args:
            parameters: Matching parameters. If None, uses default values.
        """
        self._matcher = _TemplateMatcher()
        self._parameters = parameters or MatchParameters()
        self._template = None
        self._template_size = None
        
    def set_template(self, template: np.ndarray, min_reduce_area: int = 1000) -> None:
        """
        Set the template pattern for matching.
        
        Args:
            template: Template image as numpy array (grayscale or RGB)
            min_reduce_area: Minimum template area for pyramid reduction
        """
        if not isinstance(template, np.ndarray):
            raise TypeError("Template must be a numpy array")
        
        if template.ndim == 3:
            # Convert to grayscale if needed
            template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        elif template.ndim != 2:
            raise ValueError("Template must be 2D (grayscale) or 3D (RGB) array")
        
        # Ensure uint8 type
        if template.dtype != np.uint8:
            template = (template * 255).astype(np.uint8)
        
        self._template = template
        self._template_size = template.shape
        self._matcher.learn_pattern(template, min_reduce_area)
        self._matcher.set_parameters(self._parameters._to_cpp())
    
    def match(self, image: np.ndarray) -> List[MatchResult]:
        """
        Find template matches in the given image.
        
        Args:
            image: Source image as numpy array (grayscale or RGB)
            
        Returns:
            List of MatchResult objects sorted by score (highest first)
        """
        if self._template is None:
            raise RuntimeError("Template not set. Call set_template() first.")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if image.ndim == 3:
            # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim != 2:
            raise ValueError("Image must be 2D (grayscale) or 3D (RGB) array")
        
        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Perform matching
        results = self._matcher.match(image)
        
        # Convert to MatchResult objects
        match_results = []
        for result in results:
            pos = result['position']
            match_results.append(MatchResult(
                position=(pos[0], pos[1]),
                score=float(result['score']),
                angle=float(result['angle'])
            ))
        
        return match_results
    
    def visualize_matches(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize template matches on the image.
        
        Args:
            image: Source image as numpy array
            
        Returns:
            Image with matched regions highlighted
        """
        if self._template is None:
            raise RuntimeError("Template not set. Call set_template() first.")
        
        if image.ndim == 3:
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Visualize matches
        vis_result = self._matcher.visualize_matches(image)
        
        # Convert BGR back to RGB if needed
        if len(vis_result.shape) == 3:
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
        
        return vis_result
    
    def set_parameters(self, parameters: MatchParameters) -> None:
        """Update matching parameters."""
        self._parameters = parameters
        self._matcher.set_parameters(parameters._to_cpp())
    
    @property
    def template_size(self) -> Optional[Tuple[int, int]]:
        """Get the size of the current template."""
        return self._template_size
    
    @property
    def parameters(self) -> MatchParameters:
        """Get current matching parameters."""
        return self._parameters


# Convenience functions
def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
    max_matches: int = 10,
    use_simd: bool = True
) -> List[MatchResult]:
    """
    Convenience function for quick template matching.
    
    Args:
        image: Source image as numpy array
        template: Template image as numpy array
        threshold: Score threshold for matches (0.0 to 1.0)
        max_matches: Maximum number of matches to return
        use_simd: Whether to use SIMD optimizations
        
    Returns:
        List of MatchResult objects sorted by score
    """
    params = MatchParameters(
        score_threshold=threshold,
        max_positions=max_matches,
        use_simd=use_simd
    )
    
    matcher = TemplateMatcher(params)
    matcher.set_template(template)
    
    return matcher.match(image)


def learn_pattern(template: np.ndarray, min_reduce_area: int = 1000) -> None:
    """
    Learn a pattern for template matching.
    
    This is a placeholder function for compatibility with the original C++ API.
    In Python, you typically use TemplateMatcher.set_template() instead.
    
    Args:
        template: Template image as numpy array
        min_reduce_area: Minimum template area for pyramid reduction
    """
    # This function exists for API compatibility but doesn't do anything
    # since Python wrapper handles pattern learning internally
    pass


def get_version_info() -> str:
    """Get the version information."""
    return get_version()


def check_simd_support() -> dict:
    """
    Check SIMD support on the current platform.
    
    Returns:
        Dictionary with SIMD support information
    """
    return {
        "simd_available": has_simd_support(),
        "version": get_version()
    }