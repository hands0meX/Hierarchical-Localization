import os
from pathlib import Path
import shutil
import warnings

import numpy as np
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
import argparse
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

import open3d as o3d
from open3d import io as o3d_io

import threading
import time

class HLocalizer:
    DEBUG = True
    point_cloud = o3d.geometry.PointCloud()
    colors_backup = []
    points_backup = []

    @staticmethod
    def build(dataset, overwrite):
        images = Path(f"datasets/{dataset}/")
        outputs = Path(f"outputs/{dataset}/")

        sfm_pairs = outputs / "sfm_pairs.txt"
        sfm_dir = outputs / 'sfm'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'

        feature_conf = extract_features.confs['disk']
        matcher_conf = match_features.confs['disk+lightglue']

        #展示将重建的图片
        references = [str(p.relative_to(images).as_posix()) for p in (images / "mapping/").iterdir()]
        if HLocalizer.DEBUG:
            print(references)
            print(len(references), "mapping images")
            
        # 提取特征
        extract_features.main(feature_conf, images, image_list=references, feature_path=features, as_half=False, overwrite=overwrite)
        # 图片配对
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        # 配对图片的点位匹配
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

        # 三维重建
        parseModel = reconstruction.main(overwrite=args.overwrite, sfm_dir=sfm_dir, image_dir=images, pairs=sfm_pairs, features=features, matches=matches, image_list=references)
        parseModel.export_PLY(outputs / "reconstruction.ply")
        return parseModel
    
    @staticmethod
    def detect(dataset):
        images = Path(f"datasets/{dataset}/")
        outputs = Path(f"outputs/{dataset}/")
        sfm_dir = outputs / 'sfm'

        if not (sfm_dir / 'database.db').exists():
            raise ValueError(f"Please build the sfm model first")

        loc_pairs = outputs / "pairs-loc.txt"
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'

        feature_conf = extract_features.confs['disk']
        matcher_conf = match_features.confs['disk+lightglue']
        references = [str(p.relative_to(images).as_posix()) for p in (images / "mapping/").iterdir()]
        parseModel = pycolmap.Reconstruction(str(sfm_dir))

        # 开始定位
        if not (images / "query" / "query.jpg").exists():
            os.mkdir(images / "query")

            # 找到 'mapping' 文件夹中的第一张图片
            mapping_dir = images / "mapping/"
            first_image = sorted(mapping_dir.glob('*.jpg'))[0]
            shutil.copy2(first_image, images / "query" / "query.jpg")
            warnings.warn("Please put the query images in the 'query' folder")

        query_img = "query/query.jpg"
        extract_features.main(
            feature_conf, images, image_list=[query_img], feature_path=features, overwrite=True
        )
        pairs_from_exhaustive.main(loc_pairs, image_list=[query_img], ref_list=references)
        match_features.main(
            matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
        )

        camera = pycolmap.infer_camera_from_image(images / query_img)
        ref_ids = [parseModel.find_image_with_name(r).image_id for r in references if parseModel.find_image_with_name(r) is not None]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(parseModel, conf)
        ret, log = pose_from_cluster(localizer, query_img, camera, ref_ids, features, matches)
        if HLocalizer.DEBUG:
            print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')

        pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])

        world_t_camera = pose.cam_from_world.inverse()
        rotation_camera = world_t_camera.rotation.matrix()
        translation_camera = world_t_camera.translation
        if HLocalizer.DEBUG:
            print(f"最后的位姿, 旋转矩阵: {rotation_camera}, 相机位置: {translation_camera}")
        
        if __name__ != "__main__":
            HLocalizer.point_cloud = o3d.io.read_point_cloud(f"outputs/{dataset}/reconstruction.ply")
            # 恢复原来的颜色
            # HLocalizer.point_cloud.colors = o3d.utility.Vector3dVector(HLocalizer.colors_backup)
            # 恢复原来的点位置colors_backup
            # HLocalizer.point_cloud.points = o3d.utility.Vector3dVector(HLocalizer.points_backup)

            point = np.dot(rotation_camera, [0, 0, 0]) + translation_camera
            HLocalizer.point_cloud.points.append(point)
            HLocalizer.point_cloud.colors.append([1, 0, 0])

            HLocalizer.run_visualizer()
            print("定位成功, 请查看窗口")

        return {"msg": "定位成功"}
    
    @staticmethod
    def show_window(dataset):
        if os.path.exists(f"outputs/{dataset}/reconstruction.ply"):
            HLocalizer.point_cloud = o3d.io.read_point_cloud(f"outputs/{dataset}/reconstruction.ply")
            HLocalizer.colors_backup = HLocalizer.point_cloud.colors
            HLocalizer.points_backup = HLocalizer.point_cloud.points
            print("Start the visualizer")
            threading.Thread(target=HLocalizer.run_visualizer).start()
        else:
            print("Please build the sfm model first")

    @staticmethod
    def run_visualizer():
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        time.sleep(0.1)
        vis.add_geometry(HLocalizer.point_cloud)
        while vis.poll_events():
            vis.update_geometry(HLocalizer.point_cloud)
            vis.update_renderer()
        vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFM build and detect')
    parser.add_argument('-d', '--dataset', type=str, default='desk', help='dataset name')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the previous result')
    args = parser.parse_args()

    HLocalizer.build(args.dataset, args.overwrite)
    HLocalizer.detect(args.dataset)