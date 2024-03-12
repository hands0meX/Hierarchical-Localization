import open3d as o3d
import numpy as np

# 加载点云
point_cloud = o3d.io.read_point_cloud("outputs/sacre_coeur/reconstruction.ply")

# 创建 LineSet
lines = [
    [2, 9]  # 连接第三个点和第十个点，注意索引是从0开始的
]
colors = [[1, 0, 0] for _ in range(len(lines))]  # 为每条线设置颜色，这里设置为红色
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(np.asarray(point_cloud.points)),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

# 保存 LineSet 为 PLY 文件
o3d.io.write_line_set("line_set.ply", line_set)