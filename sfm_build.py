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

argparser = argparse.ArgumentParser(description='Reconstruct a dataset with hloc.')
argparser.add_argument('-d','--dataset', type=str, default="desk", help='dataset name')
argparser.add_argument('-o', "--overwrite", action="store_true", help="use the cache of the matches")
args = argparser.parse_args()

def main():
    DEBUG = True
    dataset = args.dataset
    images = Path(f"datasets/{dataset}/")
    outputs = Path(f"outputs/{dataset}/")

    # 定义各种文件的路径
    sfm_pairs = outputs / "sfm_pairs.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs['disk']

    matcher_conf = match_features.confs['disk+lightglue']

    #展示将重建的图片
    references = [str(p.relative_to(images).as_posix()) for p in (images / "mapping/").iterdir()]
    if DEBUG:
        print(references)
        print(len(references), "mapping images")
        
    # 提取特征
    extract_features.main(feature_conf, images, image_list=references, feature_path=features, as_half=False, overwrite=args.overwrite)
    # 图片配对
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    # 配对图片的点位匹配
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # 三维重建
    parseModel = reconstruction.main(overwrite=args.overwrite, sfm_dir=sfm_dir, image_dir=images, pairs=sfm_pairs, features=features, matches=matches, image_list=references)

    # 开始定位
    if not (images / "query").exists():
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

    print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')

    pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
    

    # 读取点云
    parseModel.export_PLY(outputs / "reconstruction.ply")

    world_t_camera = pose.cam_from_world.inverse()
    rotation_camera = world_t_camera.rotation.matrix()
    translation_camera = world_t_camera.translation
    

    point_cloud = o3d_io.read_point_cloud(str(outputs / "reconstruction.ply"))

    point = np.dot(rotation_camera, [0, 0, 0]) + translation_camera
    point_cloud.points.append(point)
    point_cloud.colors.append([0, 1, 0])  # 绿色

    # o3d.visualization.draw_geometries([point_cloud])
    o3d_io.write_point_cloud(str(outputs / "reconstruction.ply"), point_cloud)

main()


