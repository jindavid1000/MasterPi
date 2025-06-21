#!/usr/bin/env python3
import cv2
import numpy as np
from typing import Tuple, List, Optional

class UniversalDistanceEstimator:
    def __init__(self):
        """
        通用距离估计器初始化
        """
        self.camera_matrix = None  # 相机内参矩阵
        self.focal_length_pixels = None  # 焦距(像素)
        self.sensor_width_mm = None  # 传感器宽度(毫米)
        self.pixel_size_mm = None  # 像素大小(毫米)
        self.image_width = None  # 图像宽度(像素)
        self.image_height = None  # 图像高度(像素)
        
    def calibrate_with_checkerboard(self, 
                                  images: List[np.ndarray],
                                  pattern_size: Tuple[int, int],
                                  square_size_mm: float) -> bool:
        """
        使用棋盘格进行相机标定
        
        参数:
            images: 棋盘格图像列表
            pattern_size: 棋盘格内角点数(宽,高)
            square_size_mm: 棋盘格方块大小(毫米)
            
        返回:
            bool: 标定是否成功
        """
        # 准备标定点
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
        objp *= square_size_mm
        
        # 图像点和物体点
        objpoints = []
        imgpoints = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                # 亚像素角点检测
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
        
        if not objpoints:
            return False
            
        # 执行相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
            
        if ret:
            self.camera_matrix = mtx
            self.focal_length_pixels = mtx[0,0]  # 假设fx=fy
            self.image_width = gray.shape[1]
            self.image_height = gray.shape[0]
            return True
        return False
    
    def set_camera_params(self,
                         focal_length_mm: float,
                         sensor_width_mm: float,
                         image_width_pixels: int,
                         image_height_pixels: int):
        """
        手动设置相机参数
        
        参数:
            focal_length_mm: 焦距(毫米)
            sensor_width_mm: 传感器宽度(毫米)
            image_width_pixels: 图像宽度(像素)
            image_height_pixels: 图像高度(像素)
        """
        self.sensor_width_mm = sensor_width_mm
        self.image_width = image_width_pixels
        self.image_height = image_height_pixels
        
        # 计算像素大小
        self.pixel_size_mm = sensor_width_mm / image_width_pixels
        
        # 计算焦距(像素)
        self.focal_length_pixels = (focal_length_mm / self.pixel_size_mm)
        
        # 构建相机矩阵
        self.camera_matrix = np.array([
            [self.focal_length_pixels, 0, image_width_pixels/2],
            [0, self.focal_length_pixels, image_height_pixels/2],
            [0, 0, 1]
        ])
    
    def estimate_distance(self,
                        object_width_mm: float,
                        pixel_width: float,
                        object_center: Optional[Tuple[float, float]] = None) -> float:
        """
        估计物体距离
        
        参数:
            object_width_mm: 物体实际宽度(毫米)
            pixel_width: 物体在图像中的像素宽度
            object_center: 物体在图像中的中心坐标(可选)
            
        返回:
            距离(毫米)
        """
        if not self.focal_length_pixels:
            raise ValueError("请先进行相机标定或设置相机参数")
            
        # 基础距离计算(小孔成像原理)
        distance_mm = (object_width_mm * self.focal_length_pixels) / pixel_width
        
        # 如果提供了物体中心位置,进行视角补偿
        if object_center is not None:
            # 计算物体中心到图像中心的偏移
            dx = object_center[0] - self.image_width/2
            dy = object_center[1] - self.image_height/2
            
            # 计算视线向量
            z = self.focal_length_pixels
            ray_length = np.sqrt(dx*dx + dy*dy + z*z)
            
            # 计算视线与光轴的夹角余弦
            cos_angle = z / ray_length
            
            # 补偿距离
            distance_mm /= cos_angle
        
        return distance_mm
    
    def estimate_distance_by_triangulation(self,
                                         object_height_mm: float,
                                         pixel_y_top: float,
                                         pixel_y_bottom: float,
                                         camera_height_mm: float) -> float:
        """
        通过三角测量估计距离(适用于已知相机高度和物体高度的情况)
        
        参数:
            object_height_mm: 物体实际高度(毫米)
            pixel_y_top: 物体顶部的y坐标(像素)
            pixel_y_bottom: 物体底部的y坐标(像素)
            camera_height_mm: 相机距地面高度(毫米)
            
        返回:
            距离(毫米)
        """
        if not self.focal_length_pixels:
            raise ValueError("请先进行相机标定或设置相机参数")
            
        # 计算物体在图像中的角度
        angle_top = np.arctan2(
            (self.image_height/2 - pixel_y_top) * self.pixel_size_mm,
            self.focal_length_pixels * self.pixel_size_mm
        )
        angle_bottom = np.arctan2(
            (self.image_height/2 - pixel_y_bottom) * self.pixel_size_mm,
            self.focal_length_pixels * self.pixel_size_mm
        )
        
        # 使用三角关系计算距离
        distance_mm = (camera_height_mm - object_height_mm) * np.tan(angle_top)
        
        return distance_mm
    
    def get_real_world_size(self,
                           pixel_size: float,
                           distance_mm: float) -> float:
        """
        根据像素大小和距离计算物体实际大小
        
        参数:
            pixel_size: 物体在图像中的像素大小
            distance_mm: 物体距离(毫米)
            
        返回:
            实际大小(毫米)
        """
        if not self.focal_length_pixels:
            raise ValueError("请先进行相机标定或设置相机参数")
            
        real_size_mm = (pixel_size * distance_mm) / self.focal_length_pixels
        return real_size_mm

def main():
    # 使用示例
    estimator = UniversalDistanceEstimator()
    
    # 方法1: 使用棋盘格标定
    # images = [cv2.imread(f"calibration_{i}.jpg") for i in range(10)]
    # estimator.calibrate_with_checkerboard(images, (9,6), 25.0)
    
    # 方法2: 手动设置参数
    estimator.set_camera_params(
        focal_length_mm=3.67,  # 典型的网络摄像头焦距
        sensor_width_mm=4.8,   # 典型的1/3英寸传感器
        image_width_pixels=1280,
        image_height_pixels=720
    )
    
    # 估计距离示例
    object_width_mm = 100
    pixel_width = 200
    distance = estimator.estimate_distance(object_width_mm, pixel_width)
    print(f"估计距离: {distance:.1f}mm")
    
    # 通过三角测量估计距离
    object_height_mm = 1000
    camera_height_mm = 2000
    pixel_y_top = 100
    pixel_y_bottom = 500
    distance = estimator.estimate_distance_by_triangulation(
        object_height_mm, pixel_y_top, pixel_y_bottom, camera_height_mm)
    print(f"三角测量距离: {distance:.1f}mm")

if __name__ == "__main__":
    main() 