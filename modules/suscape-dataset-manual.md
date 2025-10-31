---
layout: page
title: SUScape数据集使用说明
---

# SUScape数据集介绍

> 🎯 **模块目标**：了解SUScape数据集的使用

## 📊 数据集概述



## 数据格式说明

数据集以20s长度的场景为单位存储，每个场景为一个文件目录，相机图片均为jpg文件，雷达文件为pcd文件，其他为文本或者json文件。所有文件可以使用标准的工具进行查看(pcd文件可以使用meshlab或者pcl_viewer查看)。

```
>$ tree suscape_scenes/scene-000100   -d 0
suscape_scenes/scene-000100
├── aux_camera    //红外相机
│   ├── front
│   ├── front_left
│   ├── front_right               
│   ├── rear
│   ├── rear_left
│   └── rear_right
├── aux_lidar   //盲区雷达
│   ├── front
│   ├── left
│   ├── rear
│   └── right
├── calib       // 内外参标定
│   ├── aux_camera
│   │   ├── front
│   │   ├── front_left
│   │   ├── front_right
│   │   ├── rear
│   │   ├── rear_left
│   │   └── rear_right
│   ├── aux_lidar -> ../../../calib_2/aux_lidar
│   ├── camera
│   │   ├── front
│   │   ├── front_left
│   │   ├── front_right
│   │   ├── rear
│   │   ├── rear_left
│   │   └── rear_right
│   └── radar -> ../../../calib_2/radar
├── camera   //可见光相机
│   ├── front
│   ├── front_left
│   ├── front_right
│   ├── rear
│   ├── rear_left
│   └── rear_right
├── ego_pose    // gps定位信息
├── label       // 3D标注信息
├── label_fusion  // 2D标注信息
│   ├── aux_camera
│   │   ├── front
│   │   ├── front_left
│   │   ├── front_right
│   │   ├── rear
│   │   ├── rear_left
│   │   └── rear_right
│   └── camera
│       ├── front
│       ├── front_left
│       ├── front_right
│       ├── rear
│       ├── rear_left
│       └── rear_right
├── lidar       //主激光雷达点云
├── lidar_pose   // 主激光雷达位姿
├── map          // 合并点云地图
└── radar        // 毫米波雷达数据
    ├── points_front
    ├── points_front_left
    ├── points_front_right
    ├── points_rear
    ├── points_rear_left
    ├── points_rear_right
    ├── tracks_front
    ├── tracks_front_left
    ├── tracks_front_right
    ├── tracks_rear
    ├── tracks_rear_left
    └── tracks_rear_right


```

> lidar_pose为主雷达在本场景内的位置信息（以第一帧为原点）， 



## 相关论文与资源

## 🔗 导航链接

- [返回主页](../index.html)
- [下一模块：标注工具介绍](points-tool.html)
- [数据分析模块](data-analysis.html)
