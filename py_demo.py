import os
import shutil
import pycolmap
import pathlib
import numpy as np

# # 官方提取的方法
# output_path: pathlib.Path = pathlib.Path("outputs/desk/sfm")
# image_dir: pathlib.Path = pathlib.Path("datasets/desk/mapping")


# # if output_path.exists():
# #     shutil.rmtree(output_path)
# # output_path.mkdir(exist_ok=True, parents=True)
# database_path = output_path / "database.db"

# # # pycolmap.extract_features(database_path, image_dir)
# # # pycolmap.match_exhaustive(database_path)
# maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
# maps[0].summary()


# hloc方法
# print(dir(pycolmap))
# print(pycolmap.has_cuda)

# help(pycolmap.SiftExtractionOptions)
output_path: pathlib.Path = pathlib.Path("outputs/desk/sfm")
reconstruction = pycolmap.Reconstruction(output_path)
# print(reconstruction.summary())
# print(dir(reconstruction))
# print(reconstruction.num_images())
# print(reconstruction.num_images())

# image_id, image = next(iter(reconstruction.images.items()))
# print(image_id, image, image.cam_from_world.translation) # [-1.19916464  0.02789166  2.59339061]
# answer = pycolmap.absolute_pose_estimation(points2D, points3D, camera)

for image_id, image in reconstruction.images.items():
    # image.name = os.path.basename(image.name)
    # image.name = "mapping/" + image.name
    # print(image_id, dir(image))
    # print(image_id, image.summary())
    # print(image_id, image)
    # print(image_id, image.points2D)
    print(image.name, image.cam_from_world.translation)

# for point3D_id, point3D in reconstruction.points3D.items():
#     print(point3D_id, point3D)

# for camera_id, camera in reconstruction.cameras.items():
    # print(camera_id, dir(camera))
    # print(camera_id, camera.params_info)
    # print(camera_id, camera.params)
    # print(camera_id, camera.params_to_string())

# reconstruction.write(output_path)

# # 创建一个空的列表来存储cam_from_world.translation的值和对应的image name
# translations = []
# names = []

# for image_id, image in reconstruction.images.items():
#     translations.append(image.cam_from_world.translation)
#     names.append(image.name)

# # 计算所有的直线距离，并存储对应的image name
# distances = {}
# for i in range(len(translations)):
#     for j in range(i+1, len(translations)):
#         dist = np.linalg.norm(translations[i] - translations[j])
#         distances[(names[i], names[j])] = dist

# # 将结果存储在一个txt文本中
# with open('distances.txt', 'w') as f:
#     for names, dist in distances.items():
#         f.write(f'{names[0]} {names[1]} {dist}\n')