# 粒子分割分析工具 (Particle Segmentation Tool)

用于分析显微镜图像中化学粒子（晶体荧光）的计算机视觉项目。该工具能够自动检测、分割和测量图像中的粒子尺寸。

## 项目结构

```
particle-segment/
├── script/                  # 核心代码和详细文档
│   ├── README.md           # 详细代码文档
│   ├── main.py             # 主程序文件
│   ├── out_particles.json  # 输出结果文件
│   └── debug_overlay/      # 调试图像输出目录
├── samples/                # 示例数据
│   └── samples/            # 示例图像文件
│       ├── FM_t01.jpg
│       ├── FM_t10.jpg
│       └── FM_t23.jpg
└── README.md               # 项目概述（本文件）
```

## 主要开发人员

- 项目维护者：Dr. Zhang Ziyang

## 快速开始

1. 查看 `script/README.md` 获取详细使用说明
2. 运行主程序：`cd script && python main.py`

## 处理效果展示

以下是使用示例图像FM_t01.jpg的处理结果：

| 阶段 | 图像 | 说明 |
|------|------|------|
| **原图** | ![原图](script/debug_overlay/FM_t01__00_raw.png) | 原始显微镜图像 |
| **分水岭标记点** | ![标记点](script/debug_overlay/FM_t01__10_markers_n6_md60_tr0.4.png) | 检测到的粒子中心点（蓝色三角形） |
| **分水岭分割** | ![分水岭](script/debug_overlay/FM_t01__11_watershed_labels_k6.png) | 基于标记点的分割结果 |
| **边界叠加（灰度）** | ![边界灰度](script/debug_overlay/FM_t01__12_edges_on_raw.png) | 分割边界叠加在原灰度图上 |
| **边界叠加（彩色）** | ![边界彩色](script/debug_overlay/FM_t01__13_edges_on_raw_bgr.png) | 分割边界叠加在原彩色图上 |
| **最终结果** | ![最终结果](script/debug_overlay/FM_t01_boxes_alllabels.png) | 所有粒子的最小外接矩形框 |

## 许可证

本项目仅用于学术研究和教育目的。
