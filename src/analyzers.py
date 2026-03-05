# src/analyzers.py
import numpy as np
import cv2
import pandas as pd
from src import core

class SideViewAnalyzer:
    def __init__(self, crank_mm, display_w=None, display_h=None, detector=None, frames_data=None):
        self.crank_mm = crank_mm
        self.detector = detector
        self.frames_data = frames_data if frames_data is not None else []
        # 优化滤波参数：适应高并发下的跳变
        filter_keys = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                       'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                       'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                       'left_heel', 'right_heel', 'left_toe', 'right_toe']
        self.filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2), min_cutoff=0.5, beta=0.005) for k in filter_keys}
        self.side_votes = {'left': 0, 'right': 0}
        self.locked_side = None

    def process(self, frame, results, frame_idx):
        if not results or self.detector is None: return frame, False
        raw_lm = self.detector.get_landmarks_dict(results, frame.shape)
        if not raw_lm: return frame, False

        # 平滑处理
        clean_lm = {k: self.filters[k](frame_idx, np.array(v)) for k, v in raw_lm.items() if k in self.filters}
        
        # 侧边锁定逻辑
        detected = core.detect_side(clean_lm)
        if self.locked_side is None:
            self.side_votes[detected] += 1
            if frame_idx >= 30:
                self.locked_side = 'left' if self.side_votes['left'] >= self.side_votes['right'] else 'right'
            current_side = detected
        else:
            current_side = self.locked_side

        # 解析运动学角度
        unified_lm = core.get_primary_landmarks(clean_lm, current_side)
        angles = core.analyze_posture(unified_lm)

        self.frames_data.append({
            'frame_idx': frame_idx, 'angles': angles, 'landmarks': unified_lm, 'side': current_side
        })

        self._draw_visuals(frame, unified_lm, angles)
        return frame, True

    def _draw_visuals(self, frame, lm, angles):
        # 绘制生理连线
        for k1, k2 in [('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'toe')]:
            if k1 in lm and k2 in lm:
                cv2.line(frame, tuple(map(int, lm[k1])), tuple(map(int, lm[k2])), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Knee Ext: {angles.get('knee', 0):.1f}", (20, 50), 1, 1.8, (0, 255, 0), 2)

class FrontViewAnalyzer:
    def __init__(self, crank_mm=170, frames_data=None, detector=None):
        self.detector = detector
        self.frames_data = frames_data if frames_data is not None else []

    def process(self, frame, results, frame_count):
        if not results or self.detector is None: return frame, False
        lm = self.detector.get_landmarks_dict(results, frame.shape)
        if not lm: return frame, False
        
        h, w = frame.shape[:2]
        current_metrics = {}
        for side in ['left', 'right']:
            k, a, hip = f'{side}_knee', f'{side}_ankle', f'{side}_hip'
            if all(pt in lm for pt in [k, a, hip]):
                # 计算膝盖相对于脚踝的横向偏移（px）
                offset = lm[k][0] - lm[a][0]
                current_metrics[f'{side}_knee_x_offset'] = offset
                
                # 视觉反馈：力线追踪
                cv2.line(frame, (int(lm[a][0]), 0), (int(lm[a][0]), h), (255, 0, 0), 1) # 脚踝基准垂线
                cv2.line(frame, tuple(map(int, lm[hip])), tuple(map(int, lm[k])), (0, 255, 255), 2) # 髋-膝线
                cv2.circle(frame, tuple(map(int, lm[k])), 7, (0, 255, 255), -1)

        self.frames_data.append({'frame_idx': frame_count, 'angles': current_metrics})
        return frame, True