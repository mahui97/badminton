# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Tkagg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import time
import os

 # 程序流程
 # 1.准备好一系列来相机标定的图片
 # 2.对每张图片提取角点信息
 # 3.由于角点信息不够精确，进一步提取亚像素角点信息
 # 4.在图片中画出提取出的角点
 # 5.相机标定
 # 6.对标定结果评价，计算误差
 # 7.使用标定结果对原图片进行校正

path = './myImages'   # 文件路径
objp = np.zeros((11 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 20
# mgrid是meshgrid的缩写，生成的是坐标网格，输出的参数是坐标范围，得到的网格的点坐标
op = [] # 存储世界坐标系的坐标X，Y，Z，在张正友相机标定中Z=0
imgpoints = []  # 像素坐标系中角点的坐标
for i in os.listdir(path):
    #读取每一张图片
    file = '/'.join((path, i))
    a = cv2.imread(file)
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    print(file)
    # 确定输入图像中是否有棋盘格图案，并检测棋盘格的内角点
    ret, corners = cv2.findChessboardCorners(b, (11, 8), None)
    print("ret: ", ret)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret == True: # 如果所有的内角点都找到了
        corners2 = cv2.cornerSubPix(b, corners, (11, 11), (-1, -1), criteria)   # 提取亚像素角点信息
        imgpoints.append(corners2)
        op.append(objp)

print("op len: ", len(op))
print("imgpoints: ", len(imgpoints))
# 相机标定的核心
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(op, imgpoints, b.shape[::-1], None, None)
# ret为极大似然函数的最小值
# mtx是内参矩阵
# dist为相机的畸变参数矩阵
# rvecs为旋转向量
# tvecs为位移向量
print("params: ret mtx dist rvecs tvecs")
print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

tot_error = 0
for i in range(len(op)):
    imgpoints2, _ = cv2.projectPoints(op[i], rvecs[i], tvecs[i], mtx, dist) #对空间中的三维坐标点进行反向投影
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)   # 平均平方误差（重投影误差）
    tot_error += error
# print(ret, (tot_error / len(op))**0.5)
# cv2.namedWindow('winname', cv2.WINDOW_NORMAL)
# cv2.imshow('winname', a)
# cv2.waitKey(0)
"""下面是校正部分"""
h, w = a.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 1) # 校正内参矩阵
dst = cv2.undistort(a, mtx, dist, None)
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)  # 用于计算畸变映射
dst = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR)    # 把求得的映射应用到图像上
# crop the image
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# print(mapx.shape)
# print(mapy)
np.savez("outfile", mtx, dist)
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# print(roi)
plt.subplot(121), plt.imshow(a), plt.title('source')
plt.subplot(122), plt.imshow(dst), plt.title('undistorted')
plt.show()

