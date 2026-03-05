import cv2
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import core
# Switch to MediaPipe
from src import tracking_mp as tracking 
from src import report
from src import ai_report

# 全局变量用于存储点击的坐标
calibration_points = []

# src/analyze_video.py (约第 19-30 行)

def select_vertical_line(event, x, y, flags, param):
    global calibration_points
    # 修正：只使用 cv2.EVENT_LBUTTONDOWN
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        # 在图像上画出点击的点
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        if len(calibration_points) == 2:
            # 画出连线
            cv2.line(param, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
        cv2.imshow('Calibration: Click 2 points for Vertical Line', param)
        
def main():
    parser = argparse.ArgumentParser(description="NintAi Ultimate BikeFit Tool (MediaPipe)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video")
    parser.add_argument("--output_video", "-ov", type=str, help="Output video")
    parser.add_argument("--output_excel", "-oe", type=str, default="output/ultimate_data.xlsx", help="Output Excel")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    # --- 在 main 函数开头增加参数 ---
    parser.add_argument("--side", "-s", type=str, choices=['left', 'right'], help="Manual side override")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for tracking")
    # --- 新增：视角选择参数 ---
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side', help="Video view type")
    parser.add_argument("--crank_mm", type=float, default=170.0, help="Crank arm length in mm (e.g., 170, 172.5)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    report_dir = os.path.dirname(args.output_excel)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): sys.exit(1)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
        
        
    # src/analyze_video.py (约第 65-100 行)

    # ... 在 cap 初始化之后 ...
    ret, first_frame = cap.read()
    if not ret: sys.exit("Cannot read video")
    
    # 定义统一的窗口名变量，防止拼写错误
    calib_win = 'Calibration: Click 2 points for Vertical Line'
    cv2.namedWindow(calib_win, cv2.WINDOW_NORMAL)
    
    # 缩放预览图到合理大小 (540p 高度)
    display_h_calib = 540 
    display_w_calib = int(width * (display_h_calib / height))
    cv2.resizeWindow(calib_win, display_w_calib, display_h_calib)
    
    # 建立校准窗口并设置回调
    cv2.imshow(calib_win, first_frame)
    cv2.setMouseCallback(calib_win, select_vertical_line, first_frame)
    
    print("请在窗口中点击两个点来定义一条垂线。点击完后按任意键继续...")
    cv2.waitKey(0)
    
    # 安全地销毁窗口
    try:
        cv2.destroyWindow(calib_win)
    except:
        pass

    # 计算旋转角度 (逻辑保持不变)
    auto_rotate_deg = 0
    if len(calibration_points) == 2:
        p1, p2 = calibration_points[0], calibration_points[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        auto_rotate_deg = 90 - angle_deg
        if auto_rotate_deg > 90: auto_rotate_deg -= 180
        if auto_rotate_deg < -90: auto_rotate_deg += 180
        print(f"检测到倾斜，自动矫正角度: {auto_rotate_deg:.2f} 度")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ... 后续角度计算逻辑 ...

    # 计算旋转角度
    auto_rotate_deg = 0
    if len(calibration_points) == 2:
        p1, p2 = calibration_points[0], calibration_points[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # 计算该直线与水平线的夹角 (弧度)
        angle_rad = math.atan2(dy, dx)
        # 转换为角度
        angle_deg = math.degrees(angle_rad)
        
        # 我们的目标是让这条线变成垂线 (90度或 -90度)
        # 所需旋转角度 = 90 - 当前角度
        auto_rotate_deg = 90 - angle_deg
        # 修正：如果夹角接近 -90度，则目标是 -90度
        if auto_rotate_deg > 90: auto_rotate_deg -= 180
        if auto_rotate_deg < -90: auto_rotate_deg += 180
        
        print(f"检测到倾斜，自动矫正角度: {auto_rotate_deg:.2f} 度")

    # 重置视频进度到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 窗口优化：解决比例奇怪且太大的问题 ---
    window_name = 'NintAi Quad (MP)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 保持视频原始比例，缩放到显示器高度的一半
    display_h = 540 
    display_w = int(width * (display_h / height))
    cv2.resizeWindow(window_name, display_w, display_h)
    
    # --- 修复 1：先定义数据存储列表 ---
    frames_data = []

    # --- 核心重构：根据视角初始化分析器 ---
    from src import analyzers

    print("Initializing NintAi Tracking (MediaPipe Tasks)...")
    
    # --- 修复：先初始化数据列表 ---
    # 模型路径处理
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_heavy.task'))
    if not os.path.exists(model_path):
        model_path = 'src/models/pose_landmarker_heavy.task'
    
    detector = tracking.PoseDetectorMP(model_path=model_path, model_complexity=1, use_gpu=args.gpu)

    # --- 修复：现在初始化分析器，传入已经定义的 frames_data ---
    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm,
                                               display_w=display_w,
                                               display_h=display_h,
                                               detector=detector,
                                               frames_data=frames_data)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer(display_w=display_w,
                                               display_h=display_h,
                                               detector=detector,
                                               frames_data=frames_data)
    else:
        # 确认 SideViewAnalyzer 拼写正确且 analyzers.py 已保存
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm,
                                              display_w=display_w,
                                              display_h=display_h,
                                              detector=detector,
                                              frames_data=frames_data)

    # Filters
    filter_keys = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                   'left_heel', 'right_heel', 'left_toe', 'right_toe']
                   
    filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2)) for k in filter_keys}
    
    side_votes = {'left': 0, 'right': 0}
    locked_side = None
    FRAMES_TO_LOCK = 30
    frame_count = 0

    print(f"Processing... {args.input}")
    t_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 应用交互式校准的角度（反转符号以正确旋转方向）
        final_rotate = -(auto_rotate_deg + getattr(args, 'rotate_deg', 0))
        
        if final_rotate != 0:
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, final_rotate, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            
        # ... 后续 MediaPipe 处理 ...

        # timestamp for video mode
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = detector.predict(frame, timestamp_ms=ts_ms)

        # 将分析和绘制逻辑完全托管给对应的分析器
        processed_frame, is_valid_frame = analyzer.process(frame, results, frame_count)

        if processed_frame is not None:
            if out:
                out.write(processed_frame)
            cv2.imshow(window_name, processed_frame)

        # 如果分析器认为此帧有效，可以让它填充 frames_data 或其他结构
        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    
    if not frames_data: sys.exit()
    
    df = pd.DataFrame([f['angles'] for f in frames_data])
    df['frame_idx'] = [f['frame_idx'] for f in frames_data]
    df = df[df['knee'] > 0]
    if df.empty: sys.exit()
    
    # --- Report Prep ---
    idx_bdc = df['knee'].idxmax()
    vals_bdc = df.loc[idx_bdc]
    idx_tdc = df['knee'].idxmin()
    vals_tdc = df.loc[idx_tdc]
    
    facing_right = True
    if frames_data[0]['side'] == 'left': facing_right = False
    
    best_x = -1e9 if facing_right else 1e9
    idx_front = 0
    
    target_k = 'toe'
    
    # 找到脚尖最靠前的帧索引
    best_x = -1e9 if facing_right else 1e9
    target_frame_idx = -1
    
    # 直接遍历 frames_data 寻找目标 frame_idx
    for f in frames_data:
        # 只有当该帧在 df 中（即角度有效）且有脚尖数据时才计算
        if target_k in f['landmarks'] and f['angles'].get('knee', 0) > 0:
            x = f['landmarks'][target_k][0]
            if (facing_right and x > best_x) or (not facing_right and x < best_x):
                best_x = x
                target_frame_idx = f['frame_idx']

    # 使用 frame_idx 在 df 中精准定位，避免 iloc 越界
    if target_frame_idx != -1:
        vals_front = df[df['frame_idx'] == target_frame_idx].iloc[0]
    else:
        # 兜底：如果没找到，就用 TDC 帧的数据
        vals_front = vals_tdc
    
    # 提取所有有效的横向偏移数据
    x_offsets = [f['angles'].get('knee_x_offset', 0) for f in frames_data if f['angles'].get('knee_x_offset', 0) != 0]
    
    # 利用脚踝运动轨迹校准像素比例
    ankle_ys = [f['landmarks']['ankle'][1] for f in frames_data if 'ankle' in f['landmarks']]
    if ankle_ys:
        pixel_diameter = max(ankle_ys) - min(ankle_ys)
        # 1 像素等于多少毫米 = (2 * 曲柄长度) / 像素直径
        mm_per_pixel = (2 * args.crank_mm) / pixel_diameter
        print(f"Calibration: 1 pixel = {mm_per_pixel:.4f} mm (Based on Crank Circle)")
    else:
        mm_per_pixel = 1.0 # 无法校准则退回原始比例
    
    # 更新 stats 字典
    stats = {
        'knee_ext_max': df['knee'].max(),
        'knee_flex_min': df['knee'].min(),
        'hip_closed_min': df['hip'].min(),
        'back_avg': df['back'].mean(),
        'arm_avg': df['arm_torso'].mean(),
        'neck_avg': df['neck'].mean(),
        'foot_angle_avg': df.get('foot_angle', pd.Series([0])).mean(),
        # --- 新增数据 ---
        'knee_lateral_avg': np.mean(x_offsets) if x_offsets else 0,
        'knee_lateral_std': np.std(x_offsets) if x_offsets else 0
    }
    print("Generating Quad-View Snapshots (MP Tasks)...")
    
    def create_snapshot(idx, filename, title, overlay_metrics):
        c = cv2.VideoCapture(args.input)
        c.set(cv2.CAP_PROP_POS_FRAMES, frames_data[idx]['frame_idx']-1)
        _, img = c.read()
        c.release()
        if img is None: return None
        
        lm = frames_data[idx]['landmarks']
        clean_lm = frames_data[idx]['clean_lm']

        # Shoe
        if 'ankle' in lm and 'heel' in lm and 'toe' in lm:
             a = tuple(map(int, lm['ankle']))
             h = tuple(map(int, lm['heel']))
             t = tuple(map(int, lm['toe']))
             pts = np.array([a, h, t], np.int32)
             cv2.polylines(img, [pts], True, (0,255,255), 2)
             cv2.fillPoly(img, [pts], (0,100,100))

        # Skeleton
        skel = [('shoulder', 'elbow'), ('elbow', 'wrist'), 
                 ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]
        for k1, k2 in skel:
             if k1 in lm and k2 in lm:
                 p1, p2 = tuple(map(int, lm[k1])), tuple(map(int, lm[k2]))
                 cv2.line(img, p1, p2, (0,255,255), 4, cv2.LINE_AA)
        
        # Dots
        for k, v in clean_lm.items():
            if frames_data[idx]['side'] in k or 'nose' in k:
                try: cv2.circle(img, tuple(map(int, v)), 6, (0,0,255), -1)
                except: pass

        cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        y = 100
        for k, v in overlay_metrics.items():
            label = f"{k}: {v:.1f}"
            cv2.putText(img, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y += 40
            
        path = os.path.join(report_dir, filename)
        cv2.imwrite(path, img)
        return path

    # 生成富文本诊断书
    diagnostic_prompt = ai_report.generate_diagnostic_prompt(stats, locked_side or 'right')
    
    # 打印到终端方便直接复制
    print("\n" + "="*30 + " AI DIAGNOSTIC PROMPT " + "="*30)
    print(diagnostic_prompt)
    print("="*82 + "\n")
    
    # 同时保存到文件
    prompt_path = os.path.join(report_dir, "ai_diagnosis_request.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(diagnostic_prompt)
    print(f"Rich text prompt saved to: {prompt_path}")

if __name__ == "__main__":
    main()
