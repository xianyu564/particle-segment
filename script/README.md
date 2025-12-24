# 粒子分割分析工具 - 详细文档

## 功能特性

- **图像分割**: 使用OTSU阈值、形态学处理和分水岭算法进行精确分割
- **粒子测量**: 自动计算粒子的长轴(a)和短轴(c)尺寸
- **比例尺检测**: 自动识别图像中的比例尺，计算像素-微米转换
- **时间标记**: 从文件名或OCR识别时间信息
- **调试可视化**: 生成详细的处理步骤图像，便于调试和优化

## 安装依赖

```bash
pip install opencv-python numpy scipy scikit-image
```

可选依赖（用于OCR时间识别）：
```bash
pip install pytesseract
```

## 使用方法

### 基本使用

1. 将你的显微镜图像放入 `../samples/samples/` 目录
2. 运行主程序：

```bash
python main.py
```

### 自定义参数

在 `main.py` 的 `if __name__ == "__main__"` 部分可以调整以下参数：

- `IN_DIR`: 输入图像目录
- `OUT_JSON`: 输出JSON文件路径
- `DEBUG_DIR`: 调试图像输出目录（设为None关闭调试输出）
- `time_roi`: 时间标记区域 (x, y, w, h)
- `scale_roi`: 比例尺区域 (x, y, w, h)
- `margin`: 边界过滤边距

## 输出格式

程序会生成一个JSON文件，包含每个图像的分析结果：

```json
{
  "now_time": "2025-12-24T01:07:32",
  "file": "FM_t01.jpg",
  "image_shape": {
    "height_px": 1024,
    "width_px": 1024
  },
  "time_mark": {
    "seconds": 0,
    "raw": "0 sec",
    "source": "ocr_or_filename"
  },
  "scale": {
    "real_um": 5.0,
    "bar_px": 75,
    "um_per_px": 0.06666666666666667
  },
  "particles": [
    {
      "label": 1,
      "x_px": 451.77,
      "y_px": 172.98,
      "a_px": 174.61,
      "c_px": 200.95,
      "a_um": 11.64,
      "c_um": 13.40,
      "area_px": 28150,
      "angle_deg": 86.19
    }
  ]
}
```

## 图像处理流程

1. **遮罩覆盖层**: 移除图像中的时间和比例尺标记
2. **中值滤波**: 去除椒盐噪声
3. **CLAHE**: 对比度受限自适应直方图均衡化
4. **阈值分割**: OTSU或三角法阈值分割
5. **形态学处理**: 去除小连通域，桥接缺口，填充孔洞
6. **分水岭分割**: 基于距离变换的精确分割
7. **粒子测量**: 计算每个粒子的尺寸和位置

## 调试功能

启用 `DEBUG_DIR` 后，程序会在调试目录生成详细的处理步骤图像：

- `00_raw.png`: 原始图像
- `01_mask_overlays.png`: 遮罩覆盖层后
- `02_median_k3.png`: 中值滤波后
- `03_clahe_clip0.1.png`: CLAHE处理后
- `04_thr_otsu_*.png`: 阈值分割结果
- `05_rm_small_cc_*.png`: 去除小连通域后
- `06a_keep_big_*.png`: 保留大颗粒
- `06b_bridge_close_*.png`: 桥接处理后
- `07_fill_holes_*.png`: 填充孔洞后
- `08_open_k3_*.png`: 开运算后
- `09_distance.png`: 距离变换
- `10_markers_*.png`: 分水岭标记点
- `11_watershed_labels_*.png`: 分水岭分割结果
- `12_edges_on_raw.png`: 边界叠加在灰度图上
- `13_edges_on_raw_bgr.png`: 边界叠加在彩色图上

## 参数调整建议

### 对于不同类型的图像，可能需要调整：

- **阈值偏移** (`thr_offset`): 正值降低阈值，负值提高阈值
- **最小连通域面积** (`min_cc_area`): 根据粒子大小调整
- **桥接参数**: `bridge_kernel`, `bridge_iter` 控制缺口填充
- **分水岭参数**: `ws_min_distance`, `ws_threshold_rel` 控制分割精度

### ROI区域调整：

- `time_roi`: 时间标记区域，通常在左上角
- `scale_roi`: 比例尺区域，通常在右下角

## 依赖库说明

- **OpenCV**: 图像处理基础库
- **NumPy**: 数值计算
- **SciPy**: 科学计算，距离变换
- **scikit-image**: 分水岭算法
- **Pytesseract**: OCR时间识别（可选）

## 注意事项

1. 图像文件名建议使用 `FM_t{序号}.jpg` 格式，便于时间序列识别
2. 确保图像中有清晰的比例尺用于像素-微米转换
3. 对于低质量图像，可能需要调整阈值和形态学参数
4. 调试模式会生成大量中间结果图像，有助于参数调优

## 主要函数说明

### `segment_particles()`
核心分割函数，包含完整的图像处理管道。

### `measure_ac()`
测量每个分割粒子的尺寸参数。

### `process_one()`
处理单张图像的完整流程。

### `process_folder()`
批量处理文件夹中的所有图像。

## 核心算法参数

### 预处理参数
- `median_ksize`: 中值滤波核大小（默认3）
- `clahe_clip`: CLAHE对比度限制（默认0.1）
- `clahe_grid`: CLAHE网格大小（默认8x8）

### 阈值参数
- `thr_method`: 阈值方法（"otsu" 或 "triangle"）
- `thr_offset`: 阈值偏移（默认10）
- `auto_clip_white`: 是否自动调整阈值（默认True）

### 形态学参数
- `min_cc_area`: 最小连通域面积（默认800）
- `bridge_min_area`: 桥接最小面积（默认3000）
- `bridge_kernel`: 桥接核大小（默认5）
- `bridge_iter`: 桥接迭代次数（默认2）

### 分水岭参数
- `ws_min_distance`: 最小距离（默认60）
- `ws_threshold_rel`: 相对阈值（默认0.4）

## 输出文件

- `out_particles.json`: 包含所有粒子的测量结果
- `debug_overlay/`: 调试图像目录（如果启用）
- 每个输入图像对应一个带边界框的叠加图像
