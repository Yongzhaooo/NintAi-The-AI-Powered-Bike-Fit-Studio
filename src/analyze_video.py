# src/analyze_video.py

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
from src import tracking_mp as tracking 
from src import report
from src import ai_report


# 全局变量
calibration_points = []

def select_vertical_line(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(param, (x, y), 7, (0, 0, 255), -1)
        if len(calibration_points) >= 2:
            cv2.line(param, calibration_points[-2], calibration_points[-1], (0, 255, 0), 2)
        cv2.imshow('Calibration', param)

def main():
    parser = argparse.ArgumentParser(description="NintAi Ultimate BikeFit Tool")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side')
    parser.add_argument("--crank_mm", type=float, default=170.0)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--output_excel", "-oe", type=str, default="output/ultimate_data.xlsx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    report_dir = os.path.dirname(args.output_excel)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. 交互式旋转校准
    ret, first_frame = cap.read()
    calib_win = 'Calibration'
    cv2.namedWindow(calib_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(calib_win, 800, int(800 * height / width))
    cv2.imshow(calib_win, first_frame)
    cv2.setMouseCallback(calib_win, select_vertical_line, first_frame)
    
    print("请点击两个点定义【垂线】（如墙角）。点完后按任意键...")
    cv2.waitKey(0)
    cv2.destroyWindow(calib_win)

    auto_rotate_deg = 0
    if len(calibration_points) >= 2:
        p1, p2 = calibration_points[-2], calibration_points[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        # 修正：在图像坐标系下，垂线角度应为90度。
        # 如果你点击的是从上到下的线，dy为正。
        auto_rotate_deg = -90 + angle_deg
        print(f"检测到倾斜: {angle_deg:.2f}°, 补偿旋转: {auto_rotate_deg:.2f}°")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2. 初始化环境
    from src import analyzers
    frames_data = [] # 必须在初始化 analyzer 之前定义！
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_heavy.task'))
    detector = tracking.PoseDetectorMP(model_path=model_path, use_gpu=args.gpu)

    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer(frames_data=frames_data)
    else:
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm, detector=detector, frames_data=frames_data)

    window_name = 'NintAi Processing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, int(640 * height / width))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 应用校正
        if auto_rotate_deg != 0:
            M = cv2.getRotationMatrix2D((width//2, height//2), auto_rotate_deg, 1.0)
            frame = cv2.warpAffine(frame, M, (width, height))

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = detector.predict(frame, timestamp_ms=ts_ms)

        processed_frame, _ = analyzer.process(frame, results, frame_count)
        cv2.imshow(window_name, processed_frame)
        
        frame_count += 1
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # 3. 后处理与报告 (修复 KeyError)
    if not frames_data: 
        print("未检测到有效数据"); sys.exit()
    
    # 侧视图特有逻辑
    if args.view == 'side':
        df = pd.DataFrame([f['angles'] for f in frames_data])
        df['frame_idx'] = [f['frame_idx'] for f in frames_data]
        df = df[df['knee'] > 0] # 这里只在 side 模式运行
        if df.empty: sys.exit()
        
        # ... 原有的 side 报告逻辑 ...
        stats = {'knee_ext_max': df['knee'].max(), 'knee_flex_min': df['knee'].min()}
        print(f"侧面分析完成: 膝盖最大伸展角 {stats['knee_ext_max']:.1f}")
        
    elif args.view == 'front':
        # 1. 提取有效数据
        left_data = [f['angles'].get('left_knee_x_offset') for f in frames_data if 'left_knee_x_offset' in f['angles']]
        right_data = [f['angles'].get('right_knee_x_offset') for f in frames_data if 'right_knee_x_offset' in f['angles']]
        
        # 2. 计算统计量
        stats = {
            'left_valgus_avg': np.mean(left_data) if left_data else 0,
            'left_valgus_std': np.std(left_data) if left_data else 0,
            'right_valgus_avg': np.mean(right_data) if right_data else 0,
            'right_valgus_std': np.std(right_data) if right_data else 0
        }
        
        # 3. 生成 Prompt
        prompt = ai_report.generate_diagnostic_prompt(stats, 'front')
        
        # 4. 打印并保存
        print("\n" + "="*20 + " 诊断报告已生成 " + "="*20)
        print(prompt)
        
        # 自动保存到 output 文件夹
        report_path = os.path.join(report_dir, "front_view_diagnosis.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\n[系统提示] 报告已保存至: {report_path}")
        
        # 计算统计量
        stats = {
            'left_valgus_std': df['left_knee_x_offset'].std() if 'left_knee_x_offset' in df else 0,
            'right_valgus_std': df['right_knee_x_offset'].std() if 'right_knee_x_offset' in df else 0
        }
        
        print("\n" + "="*20 + " 正面分析结果 " + "="*20)
        print(f"右膝稳定性 (STD): {stats['right_valgus_std']:.2f} px")
        print(f"左膝稳定性 (STD): {stats['left_valgus_std']:.2f} px")
        
        # 调用 AI 诊断 Prompt
        prompt = ai_report.generate_diagnostic_prompt(stats, 'front')
        print(prompt)

if __name__ == "__main__":
    main()