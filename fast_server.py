import argparse
import json
import os
import time
from fastapi import FastAPI, Request
import numpy as np
import cv2
from typing import Optional

import uvicorn
from sfm_build_behavior import HLocalizer

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@app.post("/match/")
async def match(request: Request, w: int, h: int, dataset: str):
    if not os.path.exists(f"datasets/{dataset}"):
        return {"message": "还没有构建模型"}

    sum_buffer = await request.body()
    width = w
    height = h

    # 切分 ybuffer 和 uvbuffer
    ybuffer = sum_buffer[:width * height]
    uvbuffer = sum_buffer[width * height:]

    # 将 ybuffer 和 uvbuffer 转换为 NumPy 数组
    ybuffer_np = np.frombuffer(ybuffer, dtype=np.uint8)
    uvbuffer_np = np.frombuffer(uvbuffer, dtype=np.uint8)

    # 重新构造 YUV 图像
    yuv_image = np.concatenate((ybuffer_np.reshape((height, width)),
                                uvbuffer_np.reshape((height // 2, width))),
                                axis=0)

    gray_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2GRAY_420)
    gray_rotated = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
    output_file = f"datasets/{dataset}/query/query.jpg"
    cv2.imwrite(output_file, gray_rotated)

    return HLocalizer.detect(dataset)


@app.post("/sfm/")
async def match(request: Request, w: int, h: int, dataset: str, pos: str):
    if not dataset:
        return {"message": "dataset is required"}

    sum_buffer = await request.body()
    width = w
    height = h
    print(pos,'pos')

    # 切分 ybuffer 和 uvbuffer
    ybuffer = sum_buffer[:width * height]
    uvbuffer = sum_buffer[width * height:]

    # 将 ybuffer 和 uvbuffer 转换为 NumPy 数组
    ybuffer_np = np.frombuffer(ybuffer, dtype=np.uint8)
    uvbuffer_np = np.frombuffer(uvbuffer, dtype=np.uint8)

    # 重新构造 YUV 图像
    yuv_image = np.concatenate((ybuffer_np.reshape((height, width)),
                                uvbuffer_np.reshape((height // 2, width))),
                                axis=0)

    gray_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2GRAY_420)
    gray_rotated = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

    folder_path = f"datasets/{dataset}/mapping"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestemp = int(time.time())

    output_file = f"{folder_path}/frame_{timestemp}.jpg"
    cv2.imwrite(output_file, gray_rotated)

     # 在写入图片之后，追加一条pos的记录
    with open(f'datasets/{dataset}/pos.txt', 'a') as f:
        f.write(f"frame_{timestemp} {pos}" + '\n')

    return {"msg": "done"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", type=str, default="home")
    args = parser.parse_args()

    HLocalizer.show_window(args.dataset)
    uvicorn.run("fast_server:app", host="0.0.0.0", port=8000, reload=True)