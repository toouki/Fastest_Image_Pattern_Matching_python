#!/usr/bin/env python3
"""
Example usage of the Fast Image Pattern Matching library.

This script demonstrates basic template matching functionality.
"""

import numpy as np
import cv2
import sys
import os

# Add the parent directory to the path to import the library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from MatchTool_python.wrapper import (
        TemplateMatcher, 
        MatchParameters, 
        match_template,
        check_simd_support
    )
except ImportError as e:
    print(f"Failed to import MatchTool_python: {e}")
    print("Make sure the library is compiled with: pip install -e .")
    sys.exit(1)


def create_test_images():
    """Create test images for demonstration."""
    # Create a simple template (white square)
    template = np.zeros((50, 50), dtype=np.uint8)
    template[10:40, 10:40] = 255
    
    # Create a source image with multiple instances of the template
    image = np.zeros((200, 300), dtype=np.uint8)
    
    # Add the template at different locations
    image[20:70, 30:80] = template
    image[80:130, 150:200] = template
    image[120:170, 250:300] = template
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image, template


def example_basic_matching():
    """Basic template matching example."""
    print("=== Basic Template Matching Example ===")
    
    # Check SIMD support
    simd_info = check_simd_support()
    print(f"SIMD support: {simd_info['simd_available']}")
    print(f"Version: {simd_info['version']}")
    print()
    
    # Create test images
    image, template = create_test_images()
    
    # Use the convenience function
    print("Using convenience function...")
    results = match_template(image, template, threshold=0.7, max_matches=5)
    
    print(f"Found {len(results)} matches:")
    for i, result in enumerate(results):
        print(f"  Match {i+1}: position={result.position}, score={result.score:.3f}")
    print()
    
    # Use the class-based approach
    print("Using TemplateMatcher class...")
    matcher = TemplateMatcher()
    matcher.set_template(template)
    results = matcher.match(image)
    
    print(f"Found {len(results)} matches:")
    for i, result in enumerate(results):
        print(f"  Match {i+1}: position={result.position}, score={result.score:.3f}")
    print()


def example_custom_parameters():
    """Example with custom matching parameters."""
    print("=== Custom Parameters Example ===")
    
    image, template = create_test_images()
    
    # Create custom parameters
    params = MatchParameters(
        score_threshold=0.6,      # Lower threshold
        max_positions=20,        # Find more matches
        use_simd=True,           # Use SIMD optimizations
        enable_subpixel=True     # Enable sub-pixel accuracy
    )
    
    matcher = TemplateMatcher(params)
    matcher.set_template(template)
    results = matcher.match(image)
    
    print(f"Found {len(results)} matches with custom parameters:")
    for i, result in enumerate(results):
        print(f"  Match {i+1}: position={result.position}, score={result.score:.3f}")
    print()


def example_visualization():
    """Example of visualizing matches."""
    print("=== Visualization Example ===")
    
    image, template = create_test_images()
    
    matcher = TemplateMatcher()
    matcher.set_template(template)
    results = matcher.match(image)
    
    # Create visualization
    vis_image = matcher.visualize_matches(image)
    
    # Save the visualization
    output_path = "template_matches_visualization.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to: {output_path}")
    print(f"Found {len(results)} matches")
    print()


def example_real_image():
    """Example using real test images if available."""
    print("=== Real Image Example ===")
    
    # Look for test images in the Test Images directory
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Test Images")
    
    if not os.path.exists(test_dir):
        print("Test Images directory not found, skipping real image example")
        return
    
    # Find first .jpg file for template and source
    jpg_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    
    if len(jpg_files) < 2:
        print("Not enough .jpg files found in Test Images, skipping real image example")
        return
    
    # Use first two images
    template_path = os.path.join(test_dir, jpg_files[0])
    source_path = os.path.join(test_dir, jpg_files[1])
    
    try:
        # Load images
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        
        if template is None or source is None:
            print("Failed to load images, skipping real image example")
            return
        
        # Resize template if it's too large
        if template.shape[0] > 100 or template.shape[1] > 100:
            scale = min(100 / template.shape[0], 100 / template.shape[1])
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            template = cv2.resize(template, new_size)
        
        print(f"Template size: {template.shape}")
        print(f"Source size: {source.shape}")
        
        # Perform matching
        matcher = TemplateMatcher()
        matcher.set_template(template)
        results = matcher.match(source)
        
        print(f"Found {len(results)} matches:")
        for i, result in enumerate(results[:5]):  # Show top 5
            print(f"  Match {i+1}: position={result.position}, score={result.score:.3f}")
        
        if results:
            # Create and save visualization
            vis_image = matcher.visualize_matches(cv2.cvtColor(source, cv2.COLOR_GRAY2RGB))
            output_path = "real_image_matches.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing real images: {e}")
    
    print()


def benchmark_performance():
    """Benchmark the matching performance."""
    print("=== Performance Benchmark ===")
    
    import time
    
    # Create larger test images for benchmarking
    template = np.zeros((30, 30), dtype=np.uint8)
    template[5:25, 5:25] = 255
    
    image = np.random.randint(0, 50, (500, 500), dtype=np.uint8)
    # Add some template instances
    for _ in range(5):
        x, y = np.random.randint(0, 470, 2)
        image[y:y+30, x:x+30] = template
    
    # Benchmark with SIMD
    print("Benchmarking with SIMD enabled...")
    matcher_simd = TemplateMatcher(MatchParameters(use_simd=True))
    matcher_simd.set_template(template)
    
    start_time = time.time()
    results_simd = matcher_simd.match(image)
    simd_time = time.time() - start_time
    
    # Benchmark without SIMD
    print("Benchmarking with SIMD disabled...")
    matcher_no_simd = TemplateMatcher(MatchParameters(use_simd=False))
    matcher_no_simd.set_template(template)
    
    start_time = time.time()
    results_no_simd = matcher_no_simd.match(image)
    no_simd_time = time.time() - start_time
    
    print(f"Results: SIMD={len(results_simd)}, No SIMD={len(results_no_simd)}")
    print(f"Time: SIMD={simd_time:.3f}s, No SIMD={no_simd_time:.3f}s")
    if no_simd_time > 0:
        speedup = no_simd_time / simd_time
        print(f"Speedup: {speedup:.2f}x")
    print()


def main():
    """Run all examples."""
    print("Fast Image Pattern Matching - Python Examples")
    print("=" * 50)
    
    try:
        example_basic_matching()
        example_custom_parameters()
        example_visualization()
        example_real_image()
        benchmark_performance()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()