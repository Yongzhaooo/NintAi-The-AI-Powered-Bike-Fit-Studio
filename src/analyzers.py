# src/analyzers.py
import numpy as np
import cv2
import pandas as pd
from src import core

# --- 侧面分析器：继承旧版 analyze_video 逻辑 ---
class SideViewAnalyzer:
    def __init__(self, crank_mm, display_w, display_h, detector=None, frames_data=None):
        # 基本参数
        self.crank_mm = crank_mm
        self.display_w = display_w
        self.display_h = display_h
        self.detector = detector
        self.frames_data = frames_data if frames_data is not None else []

        # 过滤器与投票逻辑
        filter_keys = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                       'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                       'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                       'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                       'left_heel', 'right_heel', 'left_toe', 'right_toe']
        self.filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2)) for k in filter_keys}
        self.side_votes = {'left': 0, 'right': 0}
        self.locked_side = None
        self.FRAMES_TO_LOCK = 30

    def process(self, frame, results, frame_idx):
        # 获取原始关键点dict
        raw_lm = {}
        if self.detector is not None:
            raw_lm = self.detector.get_landmarks_dict(results, frame.shape)

        t_curr = frame_idx
        clean_lm = {}
        for k, v in raw_lm.items():
            if k in self.filters:
                clean_lm[k] = self.filters[k](t_curr, np.array(v))
            else:
                clean_lm[k] = v
        if not clean_lm:
            return frame, False

        detected = core.detect_side(clean_lm)
        if self.locked_side is None:
            self.side_votes[detected] += 1
            if frame_idx >= self.FRAMES_TO_LOCK:
                self.locked_side = 'left' if self.side_votes['left'] >= self.side_votes['right'] else 'right'
                print(f"Side Locked: {self.locked_side.upper()}")
            current_side = detected
        else:
            current_side = self.locked_side

        unified_lm = core.get_primary_landmarks(clean_lm, current_side)
        angles = core.analyze_posture(unified_lm)

        # 存储用于报告的数据
        self.frames_data.append({
            'frame_idx': frame_idx,
            'angles': angles,
            'landmarks': unified_lm,
            'clean_lm': clean_lm,
            'side': current_side
        })

        # 绘制可视化元素
        self._draw_visuals(frame, unified_lm, clean_lm, angles, current_side)
        return frame, True

    def _draw_visuals(self, frame, unified_lm, clean_lm, angles, current_side):
        # 同 analyze_video.py 中的绘制代码
        if 'ankle' in unified_lm and 'heel' in unified_lm and 'toe' in unified_lm:
            a = tuple(map(int, unified_lm['ankle']))
            h = tuple(map(int, unified_lm['heel']))
            t = tuple(map(int, unified_lm['toe']))
            pts = np.array([a, h, t], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cv2.fillPoly(frame, [pts], (0, 100, 100))
        for k, v in clean_lm.items():
            draw_it = False
            if 'nose' in k or 'eye' in k:
                draw_it = True
            elif current_side in k:
                draw_it = True
            if draw_it:
                try:
                    pt = tuple(map(int, v))
                    cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                except:
                    pass
        main_skel = [('shoulder', 'elbow'), ('elbow', 'wrist'),
                     ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]
        for k1, k2 in main_skel:
            if k1 in unified_lm and k2 in unified_lm:
                p1 = tuple(map(int, unified_lm[k1]))
                p2 = tuple(map(int, unified_lm[k2]))
                cv2.line(frame, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
        if 'knee' in angles and angles['knee'] > 0:
            needed = ['hip', 'knee', 'ankle']
            if all(k in unified_lm for k in needed):
                p1 = tuple(map(int, unified_lm['hip']))
                p2 = tuple(map(int, unified_lm['knee']))
                p3 = tuple(map(int, unified_lm['ankle']))
                core.draw_angle_arc(frame, p1, p2, p3, angles['knee'], (0, 255, 0))


# src/analyzers.py

class FrontViewAnalyzer:
    def __init__(self, crank_mm=None, display_w=None, display_h=None, detector=None, frames_data=None, is_mirrored=True):
        self.crank_mm = crank_mm
        self.display_w = display_w
        self.display_h = display_h
        self.detector = detector
        self.frames_data = frames_data if frames_data is not None else []
        self.is_mirrored = is_mirrored # 默认开启镜像处理

    def process(self, frame, results, frame_count):
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return frame, False
            
        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks[0]
        
        # 提取关键点坐标（MediaPipe 默认：0-鼻, 25-左膝, 26-右膝, 27-左踝, 28-右踝）
        # 注意：MediaPipe 识别的是生物学上的左右。如果视频是镜像的，
        # 画面左边的腿其实是人的右腿（索引26）。
        
        lm = {}
        mapping = {25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}
        for idx, name in mapping.items():
            pt = landmarks[idx]
            lm[name] = [pt.x * w, pt.y * h]

        overlay = frame.copy()
        current_frame_metrics = {'frame_idx': frame_count, 'angles': {}, 'landmarks': lm}

        for side in ['left', 'right']:
            knee = lm.get(f'{side}_knee')
            ankle = lm.get(f'{side}_ankle')
            
            if knee and ankle:
                # 计算横向偏移 (Knee Tracking)
                # 理想状态下，膝盖应该在脚踝的正上方
                x_offset = knee[0] - ankle[0]
                
                # 如果是镜像视频，我们需要反转 X 轴的偏移逻辑，以便符合“向内/向外”的直觉
                actual_offset = -x_offset if self.is_mirrored else x_offset
                
                # 记录数据
                current_frame_metrics['angles'][f'{side}_knee_x_offset'] = actual_offset
                
                # 绘制辅助线：脚踝垂直基准线
                cv2.line(overlay, (int(ankle[0]), 0), (int(ankle[0]), h), (255, 0, 0), 1)
                # 绘制膝盖点：正常绿色，内扣严重红色
                color = (0, 0, 255) if abs(actual_offset) > 40 else (0, 255, 0)
                cv2.circle(overlay, (int(knee[0]), int(knee[1])), 8, color, -1)
                
        self.frames_data.append(current_frame_metrics)
        return overlay, True
    
# --- 背面分析器：专攻骨盆稳定性 ---
class BackViewAnalyzer:
    def __init__(self, display_w=None, display_h=None, detector=None, frames_data=None, **kwargs):
        # detector/frames_data accepted for interface compatibility but unused here
        self.display_w = display_w
        self.display_h = display_h
        self.frames_data = frames_data if frames_data is not None else []

    def process(self, frame, results, frame_idx):
        if not results.pose_landmarks:
            return frame, False
            
        lm = core.tracking_mp.get_landmarks_dict(results.pose_landmarks)
        
        overlay = frame.copy()
        
        # 1. 绘制核心：骨盆水平线
        l_hip = lm.get('left_hip')
        r_hip = lm.get('right_hip')
        
        if core.valid(l_hip) and core.valid(r_hip):
            l_hip_px = tuple(map(int, l_hip[:2]))
            r_hip_px = tuple(map(int, r_hip[:2]))
            
            # 计算骨盆倾角 ( Pelvic Tilt/Rock )
            # 正常应该是水平的 (0度)
            dx = r_hip[0] - l_hip[0]
            dy = r_hip[1] - l_hip[1]
            # arccos( clip( dot / norm ) ) 
            # 或者简单点：
            tilt_angle = np.degrees(np.arctan2(dy, dx))
            
            # 在视频上绘制
            color = (0, 0, 255) if abs(tilt_angle) > 5 else (0, 255, 0) # 超过5度标红
            cv2.line(overlay, l_hip_px, r_hip_px, color, 2)
            cv2.putText(overlay, f"Tilt: {tilt_angle:.1f}deg", (l_hip_px[0], l_hip_px[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 记录数据
            self.frames_data.append({
                'frame_idx': frame_idx,
                'pelvic_tilt': tilt_angle
            })
            
        return overlay, True

    def get_final_stats(self):
        # 汇总骨盆晃动的频率和幅度
        df = pd.DataFrame(self.frames_data)
        if df.empty: return {}
        return {
            'pelvic_tilt_avg': df['pelvic_tilt'].mean(),
            'pelvic_tilt_std': df['pelvic_tilt'].std() # 标准差代表晃动幅度
        }