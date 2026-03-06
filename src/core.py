# src/core.py
import numpy as np
import cv2
import math

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.t_prev = t0
        # 确保初始化为 2 维坐标
        self.x_prev = np.array(x0[:2], dtype=float)
        self.dx_prev = np.zeros(2, dtype=float)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * 30.0)

    def __call__(self, t, x):
        # 核心修复：强制只取前两个维度 [x, y]，防止 3 维数据引发广播错误
        x = np.array(x[:2], dtype=float)
        
        t_e = t - self.t_prev
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) / t_e if t_e > 0 else np.zeros_like(x)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

def calculate_angle_horizontal(a, b):
    v = np.array(b) - np.array(a)
    return np.abs(np.degrees(np.arctan2(v[1], v[0])))

def draw_angle_arc(image, p1, p2, p3, angle, color=(0, 255, 255), radius=30):
    v1, v2 = np.array(p1) - p2, np.array(p3) - p2
    ang1, ang2 = np.degrees(np.arctan2(v1[1], v1[0])) % 360, np.degrees(np.arctan2(v2[1], v2[0])) % 360
    start, end = (ang1, ang2) if abs(ang2-ang1) < 180 else (ang2, ang1)
    cv2.ellipse(image, tuple(map(int, p2)), (radius, radius), 0, start, end, color, 2)
    cv2.putText(image, f"{int(angle)}", (int(p2[0]-15), int(p2[1]-radius-5)), 1, 1, color, 1)

def get_primary_landmarks(lm_dict, facing_side):
    """
    修改点：严格遵循分析器确定的 facing_side。
    """
    prefix = f"{facing_side}_"
    unified = {'side': facing_side}
    # 提取公共点
    for k in ['nose', 'left_ear', 'right_ear']:
        if k in lm_dict: unified[k] = lm_dict[k]
    # 提取带侧边前缀的点并重命名（如 left_knee -> knee）
    for k, v in lm_dict.items():
        if k.startswith(prefix):
            unified[k.replace(prefix, '')] = v
    return unified

def analyze_posture(lm):
    angles = {}
    # 修改 vld：允许长度为 2 (滤波后坐标) 或 3 (原始带置信度数据)
    vld = lambda pt: pt is not None and (len(pt) == 2 or len(pt) == 3)
    
    # 在所有计算中，使用 [:2] 确保只取坐标部分参与数学运算
    if vld(lm.get('hip')) and vld(lm.get('knee')) and vld(lm.get('ankle')):
        angles['knee'] = calculate_angle(lm['hip'][:2], lm['knee'][:2], lm['ankle'][:2])
    else: 
        angles['knee'] = 0
        
    if vld(lm.get('shoulder')) and vld(lm.get('hip')) and vld(lm.get('knee')):
        angles['hip'] = calculate_angle(lm['shoulder'][:2], lm['hip'][:2], lm['knee'][:2])
    else: 
        angles['hip'] = 0
        
    if vld(lm.get('shoulder')) and vld(lm.get('hip')):
        angles['back'] = calculate_angle_horizontal(lm['hip'][:2], lm['shoulder'][:2])
    else: 
        angles['back'] = 0
        
    if vld(lm.get('heel')) and vld(lm.get('toe')):
        angles['foot_angle'] = calculate_angle_horizontal(lm['heel'][:2], lm['toe'][:2])
    else: 
        angles['foot_angle'] = 0
        
    return angles