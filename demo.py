from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

DEBUG = False
images = Path("datasets/sacre_coeur")
outputs = Path("outputs/demo/")

# 定义各种文件的路径
sfm_pairs = outputs / "sfm_pairs.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

# "disk": {
#     "output": "feats-disk",
#     "model": {
#         "name": "disk",
#         "max_keypoints": 5000,
#     },
#     "preprocessing": {
#         "grayscale": False,
#         "resize_max": 1600,
#     },
# },
feature_conf = extract_features.confs['disk']
# "disk+lightglue": {
#     "output": "matches-disk-lightglue",
#     "model": {
#         "name": "lightglue",
#         "features": "disk",
#     },
# },
matcher_conf = match_features.confs['disk+lightglue']

#展示将重建的图片
references = [str(p.relative_to(images)) for p in (images / "mapping/").iterdir()]
if DEBUG:
    print(references)
    print(len(references), "mapping images")
# plot_images([read_image(images / ref) for ref in references])
    
# 提取特征
extract_features.main(feature_conf, images, image_list=references, feature_path=features, as_half=False, overwrite=False)
# 图片配对
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
# 配对图片的点位匹配
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# 三维重建
reconstruction = reconstruction.main(use_cache=True, sfm_dir=sfm_dir, image_dir=images, pairs=sfm_pairs, features=features, matches=matches, image_list=references)

# print(reconstruction.summary())
reconstruction.export_PLY(outputs / "reconstruction.ply")


