import argparse
import pycolmap
from pathlib import Path


argparser = argparse.ArgumentParser(description='Reconstruct a dataset with hloc.')
argparser.add_argument('-d','--dataset', type=str, default="desk", help='dataset name')
argparser.add_argument('-o', "--overwrite", action="store_true", help="use the cache of the matches")
args = argparser.parse_args()

dataset = args.dataset

images = Path(f"datasets/{dataset}/")
outputs = Path(f"outputs/{dataset}/sfm/")

sfm_dir = outputs / 'sfm'

pycolmap.undistort_images(sfm_dir, outputs, images)
pycolmap.patch_match_stereo(sfm_dir)  # requires compilation with CUDA
pycolmap.stereo_fusion(sfm_dir / "dense.ply", sfm_dir)