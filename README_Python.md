# Fastest Image Pattern Matching - Python Bindings

高性能图像模板匹配库的Python绑定，使用SIMD优化实现快速匹配。

## 特性

- **高性能**: 使用SIMD指令集(SSE2, AVX2, NEON)加速卷积运算
- **多尺度匹配**: 支持图像金字塔匹配
- **旋转不变**: 支持旋转模板匹配
- **亚像素精度**: 支持亚像素级定位精度
- **易于使用**: 简洁的Python API
- **OpenCV兼容**: 与OpenCV无缝集成

## 安装

1. **安装系统依赖**:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential cmake pkg-config libopencv-dev python3-dev
   
   # CentOS/RHEL/Fedora
   sudo yum install gcc-c++ cmake pkgconfig opencv-devel python3-dev
   
   # macOS
   brew install cmake pkg-config opencv python3
   ```

2. **创建虚拟环境** (推荐):

   ```bash
   python3 -m venv MatchTool_venv
   source MatchTool_venv/bin/activate
   ```

3. **安装Python依赖**:

   ```bash
   pip install -r requirements.txt
   ```

4. **编译并安装**:

   ```bash
   pip install -e .
   ```

## 快速开始

### 基本使用

```python
import numpy as np
import cv2
from MatchTool_python.wrapper import TemplateMatcher, match_template

# 加载图像
image = cv2.imread('source.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# 方法1: 使用便捷函数
results = match_template(image, template, threshold=0.8, max_matches=10)
for result in results:
    print(f"位置: {result.position}, 相似度: {result.score:.3f}")

# 方法2: 使用类接口
matcher = TemplateMatcher()
matcher.set_template(template)
results = matcher.match(image)

# 可视化结果
vis_image = matcher.visualize_matches(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
cv2.imwrite('result.jpg', vis_image)
```

### 自定义参数

```python
from MatchTool_python.wrapper import TemplateMatcher, MatchParameters

# 创建自定义参数
params = MatchParameters(
    score_threshold=0.7,      # 相似度阈值
    max_positions=20,         # 最大匹配数量
    use_simd=True,           # 使用SIMD优化
    enable_subpixel=True,    # 启用亚像素精度
    tolerance_angle=10.0     # 角度容忍度
)

# 使用自定义参数
matcher = TemplateMatcher(params)
matcher.set_template(template)
results = matcher.match(image)
```

## API 参考

### MatchParameters

匹配参数配置类:

```python
class MatchParameters:
    max_positions: int = 10          # 最大匹配数量
    max_overlap: float = 0.1         # 最大重叠率
    score_threshold: float = 0.8     # 相似度阈值
    tolerance_angle: float = 5.0     # 角度容忍度(度)
    min_reduce_area: int = 1000      # 最小缩减区域
    use_simd: bool = True           # 使用SIMD优化
    enable_subpixel: bool = False    # 亚像素精度
    fast_mode: bool = False         # 快速模式
```

### TemplateMatcher

主要模板匹配类:

```python
class TemplateMatcher:
    def __init__(self, parameters: Optional[MatchParameters] = None):
        """初始化匹配器"""
    
    def set_template(self, template: np.ndarray, min_reduce_area: int = 1000) -> None:
        """设置模板图像"""
    
    def match(self, image: np.ndarray) -> List[MatchResult]:
        """执行模板匹配"""
    
    def visualize_matches(self, image: np.ndarray) -> np.ndarray:
        """可视化匹配结果"""
```

### MatchResult

匹配结果类:

```python
class MatchResult:
    position: Tuple[float, float]  # 匹配位置 (x, y)
    score: float                   # 相似度分数 (0.0 - 1.0)
    angle: float                   # 旋转角度 (度)
```

### 便捷函数

```python
def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
    max_matches: int = 10,
    use_simd: bool = True
) -> List[MatchResult]:
    """快速模板匹配便捷函数"""

def check_simd_support() -> dict:
    """检查SIMD支持"""
```

## 示例

### 基本模板匹配

```python
import numpy as np
import cv2
from MatchTool_python.wrapper import TemplateMatcher

# 创建测试图像
template = np.zeros((50, 50), dtype=np.uint8)
template[10:40, 10:40] = 255  # 白色方形模板

image = np.zeros((200, 200), dtype=np.uint8)
image[30:80, 50:100] = template  # 在图像中放置模板
image[100:150, 120:170] = template  # 放置另一个实例

# 执行匹配
matcher = TemplateMatcher()
matcher.set_template(template)
results = matcher.match(image)

print(f"找到 {len(results)} 个匹配:")
for i, result in enumerate(results):
    print(f"匹配 {i+1}: 位置={result.position}, 相似度={result.score:.3f}")

# 可视化
vis = matcher.visualize_matches(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
cv2.imwrite('matches.jpg', vis)
```

### 性能对比

```python
import time
from MatchTool_python.wrapper import TemplateMatcher, MatchParameters

# 创建测试数据
template = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)

# SIMD优化测试
matcher_simd = TemplateMatcher(MatchParameters(use_simd=True))
matcher_simd.set_template(template)

start = time.time()
results_simd = matcher_simd.match(image)
simd_time = time.time() - start

# 无SIMD测试
matcher_no_simd = TemplateMatcher(MatchParameters(use_simd=False))
matcher_no_simd.set_template(template)

start = time.time()
results_no_simd = matcher_no_simd.match(image)
no_simd_time = time.time() - start

print(f"SIMD: {simd_time:.3f}s, 找到 {len(results_simd)} 个匹配")
print(f"无SIMD: {no_simd_time:.3f}s, 找到 {len(results_no_simd)} 个匹配")
print(f"加速比: {no_simd_time/simd_time:.2f}x")
```

运行示例:

```bash
python3 MatchTool_python/example.py
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。