from ultralytics import YOLO
import cv2
import numpy as np
import os
from rtmlib import Wholebody
# 修改 src/tracking.py
import onnxruntime as ort

class PoseDetector:
    def __init__(self, model_path='yolo11n-pose_openvino_model/'):
        """
        初始化 YOLO11 Pose 模型，使用 OpenVINO 引擎以提高 CPU 推理效率。
        """
        # 检查 OpenVINO 模型文件夹是否存在
        if not os.path.exists(model_path):
            # 如果找不到 OpenVINO 模型，回退到原始 .pt 文件（可选逻辑）
            fallback_model = 'yolo11n-pose.pt'
            if os.path.exists(fallback_model):
                print(f"警告: 未找到 OpenVINO 模型路径 {model_path}，回退到 {fallback_model}")
                self.model = YOLO(fallback_model)
            else:
                raise FileNotFoundError(f"未找到模型路径: {model_path}")
        else:
            # 加载 OpenVINO 格式模型，ultralytics 会自动调用 OpenVINO 后端
            # 这里的 task='pose' 是为了明确指定任务类型
            self.model = YOLO(model_path, task='pose')
            print(f"成功加载 OpenVINO 模型: {model_path}")

    def predict(self, image):
        """
        在视频帧上运行 YOLO11 姿态估计。
        """
        # 使用 OpenVINO 后端进行推理
        results = self.model(image, verbose=False)
        return results[0] if results else None

    def get_landmarks_dict(self, results, image_shape):
        """
        提取所有 COCO 关键点 (0-16) 并映射到命名的键 (left_*, right_*)。
        """
        h, w = image_shape[:2]
        lm_dict = {}
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return lm_dict

        # 获取第一个检测到的人的数据
        kpts = results.keypoints.data[0].cpu().numpy()
        
        # COCO 关键点映射关系
        mapping = {
            0: 'nose',
            1: 'left_eye', 2: 'right_eye',
            3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder',
            7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle'
        }
        
        for idx, name in mapping.items():
            if idx < len(kpts):
                x, y, conf = kpts[idx]
                if conf > 0.3: # 置信度阈值
                    lm_dict[name] = [x, y]
        
        # 为了兼容旧代码，将检测到的侧边映射到通用键名
        # 注意：实际分析中应根据视角动态选择 'left_' 或 'right_'
        for simple in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
            left_key = f"left_{simple}"
            if left_key in lm_dict:
                lm_dict[simple] = lm_dict[left_key]

        return lm_dict



class PoseDetectorRTM:
    def __init__(self, device='cuda', mode='lightweight'):
        # 强制指定后端优先级
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = Wholebody(
            to_openpose=False, 
            mode=mode, 
            backend='onnxruntime', 
            device=device
        )

    def predict_image(self, image):
        return self.model(image)

    def get_landmarks_dict(self, results, image_shape):
        keypoints, scores = results
        lm_dict = {}
        if len(keypoints) == 0: return lm_dict

        kpts = keypoints[0]
        conf = scores[0]

        # 映射项目所需点位 (索引参考 RTMW 133点)
        mapping = {
            0: 'nose',
            5: 'left_shoulder', 6: 'right_shoulder',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle',
            19: 'left_heel', 17: 'left_toe',
            22: 'right_heel', 20: 'right_toe'
        }
        
        for idx, name in mapping.items():
            # 核心修改：返回 [x, y, confidence]，让分析器能自动识别侧边
            lm_dict[name] = [float(kpts[idx][0]), float(kpts[idx][1]), float(conf[idx])]
        
        return lm_dict