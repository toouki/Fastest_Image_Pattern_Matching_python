"""
Fastest Image Pattern Matching Python Bindings

This package provides Python bindings for the high-performance image pattern matching
library implemented with SIMD optimizations for fast template matching.

Main features:
- Fast Normalized Cross Correlation (NCC) template matching
- SIMD-accelerated convolution operations
- Rotation-invariant pattern matching
- Multi-scale pyramid matching
- Sub-pixel accuracy estimation
"""

from .core_matcher import (
    TemplateMatcher,
    MatchResult,
    MatchParameters,
    learn_pattern,
    match_template,
    get_version,
    has_simd_support
)

__version__ = "0.1.0"
__all__ = [
    "TemplateMatcher",
    "MatchResult", 
    "MatchParameters",
    "learn_pattern",
    "match_template", 
    "get_version",
    "has_simd_support"
]