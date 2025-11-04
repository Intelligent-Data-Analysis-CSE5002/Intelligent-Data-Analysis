---
layout: page
title: SUScape数据集使用说明
---


## 下载

校内下载  [http://172.18.35.208:18088](http://172.18.35.208:18088/)


仅测试可以下载v1.0-mini部分，包含2个场景。


下载后将数据解压，目录结构如下(v1.0-mini示例)

![suscape-extracted](./suscape-dataset-images/suscape_extracted.png)

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

> 部分目录不包含在下载文件中


## 开发包安装
```
pip install numpy

git clone https://github.com/sustech-isus/suscape-devkit

cd suscape-devkit
pip install -e .

```

## 开发包使用测试


```
from suscape.dataset import SuscapeDataset, SuscapeScene, box3d_to_corners

# 加载数据集
dataset = SuscapeDataset('../suscape-test')  #  解压后的数据集根路径

# 获取所有场景名称
print(dataset.get_scene_names())

# 获取单个场景
scene = dataset.get_scene("scene-000040")

# 场景元信息
print(scene.meta['frames'])
print(scene.meta['calib']['camera']['front']['intrinsic'])
print(scene.meta['calib']['camera']['front']['lidar_to_camera'])

# 加载场景标注信息
scene.load_labels()
print(scene.labels[scene.meta['frames'][0]])


boxes = scene.get_boxes_by_frame(scene.meta['frames'][0])
print(boxes)

print(scene.get_boxes_of_obj(id="1"))

print(scene.find_box_in_frame(frame=scene.meta['frames'][0], id="1"))

# 获取内外参数
calib = scene.get_calib_for_frame("camera", "front", scene.meta['frames'][0])
lidar2cam, intrinsic = calib[0], calib[1]
print("lidar2cam:", lidar2cam)
print("intrinsic:", intrinsic)

# 获取场景内所有3d box
print(scene.list_objs())


# 读取lidar数据
print(scene.read_lidar(scene.meta['frames'][0]))

# 读取lidar pose
print(scene.read_lidar_pose(scene.meta['frames'][0]))
scene.load_lidar_pose()
print(scene.lidar_pose[scene.meta['frames'][1]])

# 3d box转为8个顶点坐标
print(box3d_to_corners(boxes[1]))



# 读取图片
# pip install opencv-python matplotlib
import matplotlib.pyplot as plt
import cv2

imgpath = scene.get_image_path("camera", "front", scene.meta['frames'][0])
img = cv2.imread(imgpath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()



# 读取lidar数据并显示
# show 3d lidar pts
# pip install open3d
import open3d as o3d
pts = scene.read_lidar(scene.meta['frames'][0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
o3d.visualization.draw_geometries([pcd])


# 将lidar点投射到图片上
# project 3d points onto image
import numpy as np
frame = scene.meta['frames'][0]
pts = scene.read_lidar(frame)
image = cv2.imread(scene.get_image_path("camera", "front", frame))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

calib = scene.get_calib_for_frame("camera", "front", frame)
lidar2cam, intrinsic = calib[0], calib[1]
# filter points in front of camera
pts_hom = np.hstack((pts[:,:3], np.ones((pts.shape[0],1))))
pts_cam = (lidar2cam @ pts_hom.T).T
pts_cam = pts_cam[pts_cam[:,2]>0]
# project
pts_2d = (intrinsic @ pts_cam[:,:3].T).T
pts_2d[:,0] /= pts_2d[:,2]
pts_2d[:,1] /= pts_2d[:,2]  

# filter those out of image
h, w, _ = image.shape
pts_2d = pts_2d[(pts_2d[:,0]>=0) & (pts_2d[:,0]<w) & (pts_2d[:,1]>=0) & (pts_2d[:,1]<h)]

for p in pts_2d:
    cv2.circle(image, (int(p[0]), int(p[1])), 1, (0,255,0), -1)
plt.imshow(image)
plt.show()



# 将3dbox投射到图像上
# draw 3dboxes on image
import numpy as np
frame = scene.meta['frames'][0]
boxes = scene.get_boxes_by_frame(frame)
image = cv2.imread(scene.get_image_path("camera", "front", frame))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for box in boxes:
    corners = box3d_to_corners(box)
    # project corners to image
    corners_hom = np.hstack((corners, np.ones((8,1))))
    corners_cam = (lidar2cam @ corners_hom.T).T
    corners_2d = (intrinsic @ corners_cam[:,:3].T).T

    # filter those behind camera
    corners_2d = corners_2d[corners_cam[:,2]>0]

    if corners_2d.shape[0] !=8:
        continue

    corners_2d[:,0] /= corners_2d[:,2]
    corners_2d[:,1] /= corners_2d[:,2]  
    corners = corners_2d[:, :2]
    for p in corners:
        cv2.circle(image, (int(p[0]), int(p[1])), 1, (0,255,0), -1)
    # draw lines
    for i,j in [(0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]:

        cv2.line(image, (int(corners[i,0]), int(corners[i,1])),
                 (int(corners[j,0]), int(corners[j,1])), (0,255,0), 2)

plt.imshow(image)
plt.show()


```
![alt text](./suscape-dataset-images/image-1.png)

![alt text](./suscape-dataset-images/image-2.png)

![alt text](./suscape-dataset-images/image-3.png)

![alt text](./suscape-dataset-images/image-4.png)


参考[demo代码](https://github.com/sustech-isus/suscape-devkit/blob/main/tests/demo.py)


## 相关论文与资源

## 🔗 导航链接

- [返回主页](../index.html)
- [下一模块：标注工具介绍](points-tool.html)
- [数据分析模块](data-analysis.html)
