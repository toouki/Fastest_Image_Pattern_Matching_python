from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys

# 获取OpenCV路径
def get_opencv_info():
    try:
        import cv2
        # 如果OpenCV已安装，从Python包获取信息
        opencv_info = {
            'include_dirs': [],
            'library_dirs': [],
            'libraries': []
        }
        
        # 尝试从cv2获取编译信息
        if hasattr(cv2, 'getBuildInformation'):
            build_info = cv2.getBuildInformation()
            # 简化版本：假设标准路径
            opencv_info.update({
                'include_dirs': ['/usr/include/opencv4', '/usr/local/include/opencv4'],
                'library_dirs': ['/usr/lib/x86_64-linux-gnu', '/usr/local/lib'],
                'libraries': ['opencv_core', 'opencv_imgproc', 'opencv_highgui']
            })
        
        return opencv_info
    except ImportError:
        pass
    
    # 尝试pkg-config
    try:
        import pkgconfig
        if pkgconfig.exists('opencv4'):
            return pkgconfig.parse('opencv4')
        elif pkgconfig.exists('opencv'):
            return pkgconfig.parse('opencv')
    except ImportError:
        pass
    except Exception:
        pass
    
    # 回退到硬编码路径
    return {
        'include_dirs': ['/usr/include/opencv4', '/usr/local/include/opencv4'],
        'library_dirs': ['/usr/lib/x86_64-linux-gnu', '/usr/local/lib'],
        'libraries': ['opencv_core', 'opencv_imgproc', 'opencv_highgui']
    }

# 获取OpenCV信息
opencv_info = get_opencv_info()

# SIMD优化扩展
simd_sources = [
    "MatchTool_python/src/simd_optimization.cpp",
]

# 核心匹配扩展
core_sources = [
    "MatchTool_python/src/core_matcher.cpp",
] + simd_sources

# 处理include_dirs
include_dirs = [
    pybind11.get_include(),
    "MatchTool",
    "MatchTool_python/src",
]
if 'include_dirs' in opencv_info:
    if isinstance(opencv_info['include_dirs'], list):
        include_dirs.extend(opencv_info['include_dirs'])
    else:
        include_dirs.append(opencv_info['include_dirs'])

# 处理libraries和library_dirs
libraries = []
library_dirs = []

if 'libraries' in opencv_info:
    if isinstance(opencv_info['libraries'], list):
        libraries.extend(opencv_info['libraries'])
    else:
        libraries.append(opencv_info['libraries'])

if 'library_dirs' in opencv_info:
    if isinstance(opencv_info['library_dirs'], list):
        library_dirs.extend(opencv_info['library_dirs'])
    else:
        library_dirs.append(opencv_info['library_dirs'])

ext_modules = [
    Pybind11Extension(
        "MatchTool_python.core_matcher",
        sources=core_sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=['-O3', '-march=native'],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="MatchTool_python",
    version="0.1.0",
    author="Fastest Image Pattern Matching",
    author_email="",
    description="Python bindings for the fastest image pattern matching library",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "opencv-python>=4.0.0",
        "numpy>=1.15.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    zip_safe=False,
)