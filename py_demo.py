import pycolmap
import pathlib

output_path: pathlib.Path = "pycolmap"
image_dir: pathlib.Path = "datasets/office/mapping"

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

py_demo.extract_features(database_path, image_dir)
py_demo.match_exhaustive(database_path)
maps = py_demo.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
py_demo.undistort_images(mvs_path, output_path, image_dir)
py_demo.patch_match_stereo(mvs_path)  # requires compilation with CUDA
py_demo.stereo_fusion(mvs_path / "dense.ply", mvs_path)