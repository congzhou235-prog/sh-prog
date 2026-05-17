import os, time
import numpy as np
import scipy
import scipy.optimize
from scipy.ndimage import label, center_of_mass
from ctypes import c_ubyte, addressof
from math import sqrt
import cv2
import gxipy as gx
import sys
import matplotlib.pyplot as plt
import argparse

# ====== Daheng 原生相机封装 ======
class _GX:
    def __init__(self, camera_index=1, exposure_us=5000.0, gain=0.0):
        self.dm = gx.DeviceManager()
        dev_num, _ = self.dm.update_all_device_list()
        if dev_num == 0:
            raise RuntimeError("No Daheng camera found.")

        self.cam = self.dm.open_device_by_index(int(camera_index))
        if self.cam is None:
            raise RuntimeError("Failed to open camera (maybe already opened by another app).")

        fc = self.cam.get_remote_device_feature_control()
        fc.get_enum_feature("PixelFormat").set("Mono10")
        fc.get_float_feature("ExposureTime").set(float(exposure_us))
        fc.get_float_feature("Gain").set(float(gain))
        print("PixelFormat:", fc.get_enum_feature("PixelFormat").get())

        self.cvt = self.dm.create_image_format_convert()
        self.cvt.set_dest_format(gx.GxPixelFormatEntry.MONO10)

        self.streaming = False

    def setAOI(self, x_start, y_start, width, height):

        fc = self.cam.get_remote_device_feature_control()

        fc.get_int_feature("Width").set(width)
        fc.get_int_feature("Height").set(height)
        fc.get_int_feature("OffsetX").set(x_start)
        fc.get_int_feature("OffsetY").set(y_start)

    def stream_on(self):
        self.cam.stream_on()
        self.streaming = True

    def stream_off(self):
        if self.streaming:
            self.cam.stream_off()
            self.streaming = False

    def captureImage(self):
        raw = self.cam.data_stream[0].get_image()
        try:
            buf_size = self.cvt.get_buffer_size_for_conversion(raw)
            out = (c_ubyte * buf_size)()
            self.cvt.convert(raw, addressof(out), buf_size, False)
            arr = np.frombuffer(out, dtype=np.uint16, count=buf_size)
            return arr.reshape(raw.frame_data.height, raw.frame_data.width)
        except Exception as e:
            # convert 失败：常见是刚开流/刚改ROI/拿到异常帧，重试即可
            arr = raw.get_numpy_array() if raw is not None else None
            return arr.reshape(raw.frame_data.height, raw.frame_data.width)


    def shutDown(self):
        try:
            self.cam.stream_off()
        except:
            pass
        try:
            self.cam.close_device()
            # gx.gx_close_lib()
        except:
            pass

if __name__ == "__main__":
    def parse_int_list(s):
        return [int(x) for x in s.split(',') if x.strip()]
    parser = argparse.ArgumentParser(description='Capture images from Daheng camera.')
    parser.add_argument('--exposure', type=float, default=5000.0, help='Exposure time in microseconds.')
    parser.add_argument('--frames', type=parse_int_list, help='Number of frames to capture.')
    parser.add_argument('--postfix', type=str, default='C', help='Postfix for file names.')
    parser.add_argument('--path', type=str, default='./hst_pic_white_star1_5', help='Base path for saving images.')
    args = parser.parse_args()

    exposure_us = args.exposure
    frames = args.frames
    postfix = args.postfix
    path = args.path
    npy_path = f'{path}/NPYs'
    png_path = f'{path}/PNGs'
    os.makedirs(npy_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)

    ven = _GX(camera_index=1, exposure_us=exposure_us, gain=0.0)
    ven.stream_off()
    ven.setAOI(0, 0, 2592, 1944)
    ven.stream_on()

    for i in frames:
        print(f'Now Capture Frame: {i}')
        img = ven.captureImage()
        print(type(img), img.dtype, img.min(), img.max())
        np.save(f"{npy_path}/frame{int(exposure_us)}_{i:03d}_{postfix}.npy", img)

        plt.figure()
        plt.imshow(img, cmap='viridis')
        plt.title(f'Captured Frame {i}')
        plt.colorbar()
        plt.savefig(f"{png_path}/frame{int(exposure_us)}_{i:03d}_{postfix}.png")

        time.sleep(1)

    ven.shutDown()