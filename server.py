from fastapi import FastAPI, Request
import numpy as np
import cv2
from typing import Optional

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@app.post("/match/")
async def match(request: Request, w: int, h: int, dataset: Optional[str] = "foo"):
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
    cv2.imwrite("outputs/desk/query.jpg", gray_image)
    return {"message": "Matched!"}
