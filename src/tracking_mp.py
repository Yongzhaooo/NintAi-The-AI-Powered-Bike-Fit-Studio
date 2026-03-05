'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\NintAi-The-AI-Powered-Bike-Fit-Studio\\src\\tracking_mp.py
Author       : Yongzhao Chen
Date         : 2026-03-05 23:38:48
LastEditTime : 2026-03-05 23:38:48
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
'''
import mediapipe as mp
import cv2
import numpy as np
import os
import time

# New Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetectorMP:
    def __init__(self, model_path='src/models/pose_landmarker_heavy.task', model_complexity=1, use_gpu=True, running_mode='VIDEO'):
        """
        Initializes MediaPipe Pose Landmarker (Tasks API).
        running_mode: 'VIDEO' for sequential processing, 'IMAGE' for parallel/independent frames.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download it.")

        # 转换运行模式字符串为 MediaPipe 常量
        mode = vision.RunningMode.IMAGE if running_mode == 'IMAGE' else vision.RunningMode.VIDEO

        try:
            delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
            base_options = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
            
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mode, # 使用动态模式
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            if use_gpu:
                print(f"Warning: GPU acceleration failed ({e}). Falling back to CPU...")
                base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mode,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False
                )
                self.landmarker = vision.PoseLandmarker.create_from_options(options)
            else:
                raise e

    def predict(self, image, timestamp_ms=None):
        """用于 VIDEO 模式的检测"""
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def predict_image(self, image):
        """用于 IMAGE 模式的独立检测（并行处理必备）"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.landmarker.detect(mp_image) # detect 用于独立图像，不要求时间戳顺序

    def get_landmarks_dict(self, results, image_shape):
        """提取关键点并映射到字典"""
        if not results or not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return {}
        h, w = image_shape[:2]
        lm_dict = {}
        landmarks = results.pose_landmarks[0]
        
        mapping = {
            0: 'nose', 2: 'left_eye', 5: 'right_eye', 7: 'left_ear', 8: 'right_ear',
            11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel', 31: 'left_toe', 32: 'right_toe'
        }
        
        for idx, name in mapping.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    lm_dict[name] = [lm.x * w, lm.y * h]
        return lm_dict