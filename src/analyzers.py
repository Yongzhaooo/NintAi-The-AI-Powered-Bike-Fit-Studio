import numpy as np
import cv2
from src import core

class SideViewAnalyzer:
    def __init__(self, crank_mm=170.0, frames_data=None, target_side=None, **kwargs):
        self.crank_mm = crank_mm
        self.frames_data = frames_data if frames_data is not None else []
        self.locked_side = target_side
        self.side_votes = {'left': 0, 'right': 0}
        # 滤波器针对统一后的 Key
        filter_keys = ['shoulder', 'hip', 'knee', 'ankle', 'toe', 'heel', 'elbow', 'wrist', 'nose']
        self.filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2)) for k in filter_keys}

    def process(self, frame, lm_dict, frame_idx):
        if not lm_dict: return frame, False
        
        # 1. 100 帧简易投票锁定逻辑
        if self.locked_side is None:
            if 'left_knee' in lm_dict: self.side_votes['left'] += 1
            if 'right_knee' in lm_dict: self.side_votes['right'] += 1
            # 投票期间，哪边点多用哪边
            current_side = 'left' if lm_dict.get('left_knee') else 'right'
            # 满 100 帧后正式大选锁定
            if frame_idx >= 100:
                self.locked_side = 'left' if self.side_votes['left'] >= self.side_votes['right'] else 'right'
                print(f"\n[系统] 100帧投票完成，锁定侧边: {self.locked_side.upper()}")
        else:
            current_side = self.locked_side

        # 2. 统一化与滤波
        unified_lm = core.get_primary_landmarks(lm_dict, current_side)
        clean_lm = {}
        for k, v in unified_lm.items():
            if k in self.filters:
                clean_lm[k] = self.filters[k](frame_idx, np.array(v))
            else: clean_lm[k] = v

        # 3. 计算与存储
        angles = core.analyze_posture(clean_lm)
        self.frames_data.append({'frame_idx': frame_idx, 'angles': angles, 'side': current_side})
        
        # 4. 绘图
        knee_angle = angles.get('knee', 0)
        color = (0, 255, 0) if 140 <= knee_angle <= 150 else (0, 0, 255)
        skel = [('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle'), ('shoulder', 'elbow')]
        for k1, k2 in skel:
            if k1 in clean_lm and k2 in clean_lm:
                cv2.line(frame, tuple(map(int, clean_lm[k1])), tuple(map(int, clean_lm[k2])), color, 3)
        if all(k in clean_lm for k in ['hip', 'knee', 'ankle']) and knee_angle > 0:
            core.draw_angle_arc(frame, clean_lm['hip'], clean_lm['knee'], clean_lm['ankle'], knee_angle)
        
        cv2.putText(frame, f"SIDE: {current_side.upper()} | Knee: {knee_angle:.1f}", (20, 50), 1, 1.5, color, 2)
        return frame, True
        
class FrontViewAnalyzer:
    def __init__(self, crank_mm=170.0, frames_data=None, **kwargs):
        """ 显式定义构造函数，解决 TypeError """
        self.crank_mm = crank_mm
        self.frames_data = frames_data if frames_data is not None else []

    def process(self, frame, lm_dict, frame_count):
        if not lm_dict: return frame, False
        
        h, w = frame.shape[:2]
        current_metrics = {}
        for side in ['left', 'right']:
            k, a = f'{side}_knee', f'{side}_ankle'
            if k in lm_dict and a in lm_dict:
                offset = lm_dict[k][0] - lm_dict[a][0]
                current_metrics[f'{side}_knee_x_offset'] = offset
                
                # 视觉反馈
                cv2.circle(frame, tuple(map(int, lm_dict[a])), 10, (255, 0, 0), -1)
                color = (0, 0, 255) if abs(offset) > 40 else (0, 255, 0)
                cv2.circle(frame, tuple(map(int, lm_dict[k])), 8, color, -1)
                cv2.line(frame, (int(lm_dict[a][0]), 0), (int(lm_dict[a][0]), h), (255, 0, 0), 1)

        self.frames_data.append({'frame_idx': frame_count, 'angles': current_metrics})
        return frame, True