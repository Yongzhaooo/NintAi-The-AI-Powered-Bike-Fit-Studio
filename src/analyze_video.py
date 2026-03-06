import cv2
import argparse
import sys
import os
import math
import time
import pandas as pd
import multiprocessing as mp
from queue import Empty

# --- 路径兼容修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import tracking_mp as tracking
from src import analyzers
from src import ai_report

def pose_worker(input_queue, output_queue, model_path, use_gpu):
    """ 工作进程：仅执行推理并返回关键点 """
    detector = tracking.PoseDetectorMP(model_path=model_path, use_gpu=use_gpu, running_mode='IMAGE')
    while True:
        try:
            task = input_queue.get(timeout=2)
            if task is None: break
            
            frame_idx, frame = task
            results = detector.predict_image(frame)
            lm_dict = detector.get_landmarks_dict(results, frame.shape)
            output_queue.put((frame_idx, lm_dict))
        except Empty:
            continue
        except Exception as e:
            print(f"Worker Error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="NintAi Pro BikeFit Studio (Optimized)")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side')
    parser.add_argument("--crank_mm", type=float, default=170.0)
    parser.add_argument("--gpu", action="store_true", default=False)
    # 新增 Headless 模式参数
    parser.add_argument("--headless", action="store_true", help="不显示实时画面，最大化处理速度")
    parser.add_argument("--rotate", type=float, default=0, help="Headless 模式下的手动旋转角度")
    args = parser.parse_args()

    report_dir = os.path.join(project_root, "output")
    os.makedirs(report_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    auto_rotate_deg = args.rotate

    # --- 1. 交互式校准（仅在非 Headless 模式下运行） ---
    if not args.headless:
        ret, first_frame = cap.read()
        if ret:
            global calibration_points
            calibration_points = []
            def select_line(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    calibration_points.append((x, y))
                    cv2.circle(param, (x, y), 7, (0, 0, 255), -1)
                    if len(calibration_points) >= 2:
                        cv2.line(param, calibration_points[-2], calibration_points[-1], (0, 255, 0), 2)
                    cv2.imshow('Calibration', param)

            cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
            cv2.imshow('Calibration', first_frame)
            cv2.setMouseCallback('Calibration', select_line, first_frame)
            print("Headless 模式关闭：请点击两点定义垂线，按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyWindow('Calibration')

            if len(calibration_points) >= 2:
                p1, p2 = calibration_points[-2], calibration_points[-1]
                angle_deg = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                auto_rotate_deg = angle_deg - 90
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        print(f"Headless 模式开启：跳过交互校准。使用预设旋转角度: {auto_rotate_deg}°")

    # --- 2. 启动多进程 ---
    num_workers = max(1, mp.cpu_count() - 1)
    input_queue = mp.Queue(maxsize=num_workers * 2)
    output_queue = mp.Queue()
    
    model_path = os.path.abspath(os.path.join(current_dir, 'models/pose_landmarker_heavy.task'))
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=pose_worker, args=(input_queue, output_queue, model_path, args.gpu))
        p.start()
        processes.append(p)

    # --- 3. 初始化分析器 ---
    frames_data = [] 
    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer(frames_data=frames_data)
    else:
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)

    if not args.headless:
        window_name = 'NintAi Processing'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- 4. 主循环与性能统计 ---
    raw_frames_cache = {}
    results_cache = {}
    submitted_idx, processed_idx = 0, 0
    video_done = False
    
    start_time = time.time() # 统计开始时间

    try:
        while processed_idx < submitted_idx or not video_done:
            # 生产者
            if not video_done and input_queue.qsize() < num_workers:
                ret, frame = cap.read()
                if not ret:
                    video_done = True
                    for _ in range(num_workers): input_queue.put(None)
                else:
                    if auto_rotate_deg != 0:
                        M = cv2.getRotationMatrix2D((width // 2, height // 2), auto_rotate_deg, 1.0)
                        frame = cv2.warpAffine(frame, M, (width, height))
                    
                    raw_frames_cache[submitted_idx] = frame
                    input_queue.put((submitted_idx, frame))
                    submitted_idx += 1

            # 消费者：收集结果
            while not output_queue.empty():
                f_idx, lm_dict = output_queue.get_nowait()
                results_cache[f_idx] = lm_dict

            # 处理并显示（或仅处理）
            while processed_idx in results_cache:
                lm_dict = results_cache.pop(processed_idx)
                p_frame = raw_frames_cache.pop(processed_idx)
                
                # 在 Headless 模式下，analyzer.process 依然执行计算，但不进行 imshow
                processed_frame, _ = analyzer.process(p_frame, lm_dict, processed_idx)
                
                if not args.headless:
                    cv2.imshow(window_name, processed_frame)
                    if cv2.waitKey(1) == ord('q'):
                        video_done = True
                        break
                
                processed_idx += 1
                # 打印进度
                if processed_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = processed_idx / elapsed
                    print(f"\r进度: {processed_idx}/{total_frames if total_frames>0 else '?'} | 速度: {fps:.2f} FPS", end="")

    finally:
        total_time = time.time() - start_time
        print(f"\n\n处理完成！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均速度: {processed_idx / total_time:.2f} FPS")
        
        for p in processes:
            p.terminate()
            p.join()
        cap.release()
        cv2.destroyAllWindows()

    # --- 5. 报告生成 ---
    if not frames_data:
        print("未检测到有效数据，取消报告生成。")
        return

    # 此处逻辑与原版保持一致
    if args.view == 'side':
        df_side = pd.DataFrame([{'frame_idx': f['frame_idx'], **f['angles']} for f in frames_data if f['angles'].get('knee', 0) > 0])
        if not df_side.empty:
            stats = {
                'knee_ext_max': df_side['knee'].max(), 'knee_flex_min': df_side['knee'].min(),
                'hip_closed_min': df_side['hip'].min(), 'back_avg': df_side['back'].mean(),
                'foot_angle_avg': df_side['foot_angle'].mean()
            }
            prompt = ai_report.generate_diagnostic_prompt(stats, 'side')
            with open(os.path.join(report_dir, "side_view_diagnosis.md"), "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"侧面诊断报告已生成。已处理 {processed_idx} 帧。")

if __name__ == "__main__":
    mp.freeze_support()
    main()