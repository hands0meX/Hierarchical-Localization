from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
import open3d as o3d
import argparse
import open3d as o3d

argparser = argparse.ArgumentParser(description='Reconstruct a dataset with hloc.')
argparser.add_argument('-d','--dataset', type=str, default="outer", help='dataset name')
argparser.add_argument('-o', "--overwrite", action="store_true", help="use the cache of the matches")
print(argparser.parse_args().overwrite)

def main():
    DEBUG = True
    dataset = argparser.parse_args().dataset
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
    extract_features.main(feature_conf, images, image_list=references, feature_path=features, as_half=False, overwrite=argparser.parse_args().overwrite)
    # 图片配对
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    # 配对图片的点位匹配
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # print("features:", features)
    # 三维重建
    parseModel = reconstruction.main(overwrite=argparser.parse_args().overwrite, sfm_dir=sfm_dir, image_dir=images, pairs=sfm_pairs, features=features, matches=matches, image_list=references)

    # print(parseModel.summary())
    # print(parseModel.points3D)
    parseModel.export_PLY(outputs / "reconstruction.ply")

    # 读取点云
    pcd = o3d.io.read_point_cloud(str(outputs / "reconstruction.ply"))
    o3d.visualization.draw_geometries([pcd])

main()


