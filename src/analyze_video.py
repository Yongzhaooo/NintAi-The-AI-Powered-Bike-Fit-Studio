import cv2
import argparse
import sys
import os
import math
import time
import pandas as pd  # 必须导入，用于报告数据处理
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# --- 路径兼容修复：确保在 Windows 下能正确识别 src 包 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import tracking_mp as tracking
from src import analyzers
from src import ai_report

# 全局变量用于鼠标回调
calibration_points = []
worker_detector = None

def select_vertical_line(event, x, y, flags, param):
    """ 处理鼠标点击，定义垂线 """
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(param, (x, y), 7, (0, 0, 255), -1)
        if len(calibration_points) >= 2:
            cv2.line(param, calibration_points[-2], calibration_points[-1], (0, 255, 0), 2)
        cv2.imshow('Calibration', param)

def init_worker(model_path, use_gpu):
    """ 子进程初始化模型 """
    global worker_detector
    worker_detector = tracking.PoseDetectorMP(model_path=model_path, use_gpu=use_gpu, running_mode='IMAGE')

def process_frame_worker(frame, frame_idx, rotate_deg):
    """ 子进程执行旋转和推理，并返回可序列化的字典 """
    global worker_detector
    if rotate_deg != 0:
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_deg, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))

    results = worker_detector.predict_image(frame)
    # 核心修复：将 C++ 对象转为 Python 字典以支持跨进程传输
    lm_dict = worker_detector.get_landmarks_dict(results, frame.shape)
    return frame_idx, frame, lm_dict

def main():
    parser = argparse.ArgumentParser(description="NintAi Pro BikeFit Studio")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side')
    parser.add_argument("--crank_mm", type=float, default=170.0)
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    # 设置报告保存目录
    report_dir = os.path.join(project_root, "output")
    os.makedirs(report_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- 1. 交互式旋转校准 ---
    ret, first_frame = cap.read()
    if not ret: 
        print("无法读取视频文件"); return
    
    calib_win = 'Calibration'
    cv2.namedWindow(calib_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(calib_win, 800, int(800 * height / width))
    cv2.imshow(calib_win, first_frame)
    cv2.setMouseCallback(calib_win, select_vertical_line, first_frame)
    print("请在窗口中点击两点定义【垂线】。完成后按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyWindow(calib_win)

    auto_rotate_deg = 0
    if len(calibration_points) >= 2:
        p1, p2 = calibration_points[-2], calibration_points[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        auto_rotate_deg = angle_deg - 90 
        print(f"自动矫正角度: {auto_rotate_deg:.2f}°")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 2. 初始化环境 ---
    frames_data = [] 
    model_path = os.path.abspath(os.path.join(current_dir, 'models/pose_landmarker_heavy.task'))
    
    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer(frames_data=frames_data)
    else:
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)

    window_name = 'NintAi Processing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, int(800 * height / width))

    # --- 3. 并行处理引擎 ---
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    results_cache = {}
    next_frame_idx, submitted_count = 0, 0
    video_fully_read = False # 视频读取完毕标志位

    try:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(model_path, args.gpu)) as executor:
            futures = []
            # 循环条件：视频未读完 OR 还有任务在执行 OR 还有缓存没显示
            while not video_fully_read or futures or results_cache:
                # 生产者：提交任务
                while not video_fully_read and len(futures) < num_workers * 2:
                    ret, frame = cap.read()
                    if not ret:
                        video_fully_read = True
                        break
                    futures.append(executor.submit(process_frame_worker, frame, submitted_count, auto_rotate_deg))
                    submitted_count += 1

                # 消费者：收集结果
                done_list = [f for f in futures if f.done()]
                for f in done_list:
                    try:
                        f_idx, p_frame, lm_dict = f.result()
                        results_cache[f_idx] = (p_frame, lm_dict)
                    except Exception as e:
                        print(f"帧处理异常: {e}")
                    futures.remove(f)

                # 按序显示
                while next_frame_idx in results_cache:
                    p_frame, lm_dict = results_cache.pop(next_frame_idx)
                    processed_frame, _ = analyzer.process(p_frame, lm_dict, next_frame_idx)
                    cv2.imshow(window_name, processed_frame)
                    next_frame_idx += 1

                if cv2.waitKey(1) == ord('q'):
                    print("\n用户按 'q' 停止处理。")
                    break
    except KeyboardInterrupt:
        print("\n检测到用户中断 (Ctrl+C)，正在尝试保存已处理的数据...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"引擎已关闭。已处理帧数: {next_frame_idx}")

    # --- 4. 报告生成 ---
    if not frames_data:
        print("未检测到有效姿态数据，取消报告生成。"); sys.exit()

    report_path = ""
    prompt = ""

    if args.view == 'front':
        df_front = pd.DataFrame([f['angles'] for f in frames_data])
        stats = {
            'left_valgus_std': df_front['left_knee_x_offset'].std() if 'left_knee_x_offset' in df_front.columns else 0,
            'right_valgus_std': df_front['right_knee_x_offset'].std() if 'right_knee_x_offset' in df_front.columns else 0,
            'left_valgus_avg': df_front['left_knee_x_offset'].mean() if 'left_knee_x_offset' in df_front.columns else 0,
            'right_valgus_avg': df_front['right_knee_x_offset'].mean() if 'right_knee_x_offset' in df_front.columns else 0
        }
        
        print("\n" + "="*30 + " 膝盖稳定性分析 (正面) " + "="*30)
        print(f"右膝稳定性 (STD): {stats['right_valgus_std']:.2f} px")
        print(f"左膝稳定性 (STD): {stats['left_valgus_std']:.2f} px")
        
        prompt = ai_report.generate_diagnostic_prompt(stats, 'front')
        report_path = os.path.join(report_dir, "front_view_diagnosis.md")

    elif args.view == 'side':
        # build dataframe preserving frame_idx alongside angles
        df_side = pd.DataFrame([
            {'frame_idx': f['frame_idx'], **f['angles']}
            for f in frames_data if f['angles'].get('knee', 0) > 0
        ])
        # drop early frames (warm‑up) if column exists
        if 'frame_idx' in df_side.columns:
            df_side = df_side[df_side['frame_idx'] >= 100]
        if df_side.empty: 
            print("侧面关键角度数据不足"); sys.exit()

        stats = {
            'knee_ext_max': df_side['knee'].max() if 'knee' in df_side.columns else 0,
            'knee_flex_min': df_side['knee'].min() if 'knee' in df_side.columns else 0,
            'hip_closed_min': df_side['hip'].min() if 'hip' in df_side.columns else 0,
            'back_avg': df_side['back'].mean() if 'back' in df_side.columns else 0,
            'foot_angle_avg': df_side['foot_angle'].mean() if 'foot_angle' in df_side.columns else 0
        }
        
        print("\n" + "="*30 + " 侧面生物力学深度分析 " + "="*30)
        print(f"膝盖最大伸展角: {stats['knee_ext_max']:.1f}°")
        print(f"髋部最小闭合角: {stats['hip_closed_min']:.1f}°")
        
        prompt = ai_report.generate_diagnostic_prompt(stats, 'side')
        report_path = os.path.join(report_dir, "side_view_diagnosis.md")

    # 统一保存报告
    if report_path and prompt:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\n[系统提示] 详细诊断报告已保存至: {report_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows 多进程必须包含此行
    main()