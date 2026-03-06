'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\NintAi-The-AI-Powered-Bike-Fit-Studio\\src\\analyzers.py
Author       : Yongzhao Chen
Date         : 2026-03-06 18:16:04
LastEditTime : 2026-03-06 18:35:39
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
'''
# src/analyzers.py

import numpy as np
import cv2
from src import core

class SideViewAnalyzer:
    def __init__(self, crank_mm=170.0, frames_data=None, target_side=None, **kwargs):
        self.crank_mm = crank_mm
        self.frames_data = frames_data if frames_data is not None else []
        self.locked_side = target_side
        self.side_votes = {'left': 0, 'right': 0}
        # 过滤器 Key 需包含脚部
        filter_keys = ['shoulder', 'hip', 'knee', 'ankle', 'toe', 'heel', 'elbow', 'wrist', 'nose']
        self.filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2)) for k in filter_keys}

    def process(self, frame, lm_dict, frame_idx):
        if not lm_dict: return frame, False
        
        # 1. 100 帧投票逻辑
        if self.locked_side is None:
            # use confidence values (RTM returns [x,y,conf]) when available
            left_conf = lm_dict.get('left_knee',[0,0,0])[2] if 'left_knee' in lm_dict else 0
            right_conf = lm_dict.get('right_knee',[0,0,0])[2] if 'right_knee' in lm_dict else 0
            if left_conf > right_conf:
                self.side_votes['left'] += 1
            elif right_conf > left_conf:
                self.side_votes['right'] += 1
            # choose which side appears stronger this frame
            current_side = 'left' if left_conf >= right_conf else 'right'
            if frame_idx >= 100:
                self.locked_side = 'left' if self.side_votes['left'] >= self.side_votes['right'] else 'right'
                print(f"\n[系统] 锁定侧边: {self.locked_side.upper()}")
        else:
            current_side = self.locked_side

        # 2. 统一化与滤波
        unified_lm = core.get_primary_landmarks(lm_dict, current_side)
        clean_lm = {}
        for k, v in unified_lm.items():
            if k in self.filters:
                clean_lm[k] = self.filters[k](frame_idx, np.array(v))
            else: clean_lm[k] = v

        # 3. 计算
        angles = core.analyze_posture(clean_lm)
        self.frames_data.append({'frame_idx': frame_idx, 'angles': angles, 'side': current_side})
        
        # 4. 绘制：包含脚部三角
        knee_angle = angles.get('knee', 0)
        color = (0, 255, 0) if 140 <= knee_angle <= 150 else (0, 0, 255)
        
        # 躯干与下肢
        for k1, k2 in [('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]:
            if k1 in clean_lm and k2 in clean_lm:
                cv2.line(frame, tuple(map(int, clean_lm[k1])), tuple(map(int, clean_lm[k2])), color, 3)
        
        # --- 找回脚部三角 ---
        if all(k in clean_lm for k in ['ankle', 'heel', 'toe']):
            pts = np.array([clean_lm['ankle'], clean_lm['heel'], clean_lm['toe']], np.int32)
            cv2.fillPoly(frame, [pts], (0, 100, 100)) # 暗色填充
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2) # 亮色轮廓
        
        if all(k in clean_lm for k in ['hip', 'knee', 'ankle']) and knee_angle > 0:
            core.draw_angle_arc(frame, clean_lm['hip'], clean_lm['knee'], clean_lm['ankle'], knee_angle)
            
        cv2.putText(frame, f"SIDE: {current_side.upper()} | Knee: {knee_angle:.1f}", (20, 50), 1, 1.5, color, 2)
        return frame, True

# src/analyzers.py (FrontViewAnalyzer 优化部分)
class FrontViewAnalyzer:
    def __init__(self, crank_mm=170.0, frames_data=None, **kwargs):
        self.frames_data = frames_data if frames_data is not None else []

    def process(self, frame, lm_dict, frame_idx):
        if not lm_dict: return frame, False
        h, w = frame.shape[:2]
        current_metrics = {}
        
        for side in ['left', 'right']:
            k, a = f'{side}_knee', f'{side}_ankle'
            if k in lm_dict and a in lm_dict:
                # 仅取 X 坐标计算
                offset = lm_dict[k][0] - lm_dict[a][0]
                current_metrics[f'{side}_knee_x_offset'] = offset
                
                # 绘图：tuple(map(int, lm_dict[a][:2])) 确保只取前两个值
                ankle_pt = tuple(map(int, lm_dict[a][:2]))
                knee_pt = tuple(map(int, lm_dict[k][:2]))
                
                cv2.circle(frame, ankle_pt, 10, (255, 0, 0), -1)
                color = (0, 0, 255) if abs(offset) > 40 else (0, 255, 0)
                cv2.circle(frame, knee_pt, 8, color, -1)
                cv2.line(frame, (ankle_pt[0], 0), (ankle_pt[0], h), (255, 0, 0), 1)

        self.frames_data.append({'frame_idx': frame_idx, 'angles': current_metrics})
        return frame, True