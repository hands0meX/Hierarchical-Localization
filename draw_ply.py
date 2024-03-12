import open3d as o3d
import numpy as np

# 加载点云
point_cloud = o3d.io.read_point_cloud("outputs/desk/reconstruction.ply")
points = np.asarray(point_cloud.points)
print("获取的点数:", len(points))
print(points[0:1000])