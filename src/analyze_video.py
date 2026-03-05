import cv2
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
import math
import threading
from queue import Queue
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import core
from src import tracking_mp as tracking 
from src import report
from src import ai_report
from src import analyzers

# 全局变量
calibration_points = []
worker_detector = None  # 用于子进程缓存模型

def select_vertical_line(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(param, (x, y), 7, (0, 0, 255), -1)
        if len(calibration_points) >= 2:
            cv2.line(param, calibration_points[-2], calibration_points[-1], (0, 255, 0), 2)
        cv2.imshow('Calibration', param)

# --- 子进程初始化与工作函数 ---
def init_worker(model_path, use_gpu):
    """
    进程池初始化函数：每个子进程启动时仅运行一次，缓存模型实例。
    """
    global worker_detector
    # 并行处理必须强制使用 IMAGE 模式
    worker_detector = tracking.PoseDetectorMP(
        model_path=model_path, 
        use_gpu=use_gpu, 
        running_mode='IMAGE'
    )

def process_frame_worker(frame, frame_idx, rotate_deg):
    """
    子进程任务：利用缓存的模型处理帧。
    """
    global worker_detector
    
    # 1. 旋转校正
    h, w = frame.shape[:2]
    if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_deg, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))

    # 2. 推理 (直接使用缓存好的 worker_detector)
    results = worker_detector.predict_image(frame)
    return frame_idx, frame, results

class FileVideoStream:
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            if not self.queue.full():
                (ret, frame) = self.stream.read()
                if not ret:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def more(self):
        return self.queue.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    parser = argparse.ArgumentParser(description="NintAi Parallel Pro")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side')
    parser.add_argument("--crank_mm", type=float, default=170.0)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--output_excel", "-oe", type=str, default="output/ultimate_data.xlsx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    report_dir = os.path.dirname(args.output_excel)

    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 1. 交互式旋转校准
    ret, first_frame = cap.read()
    calib_win = 'Calibration'
    cv2.namedWindow(calib_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(calib_win, 800, int(800 * height / width))
    cv2.imshow(calib_win, first_frame)
    cv2.setMouseCallback(calib_win, select_vertical_line, first_frame)
    print("请点击两点定义垂线。按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyWindow(calib_win)

    auto_rotate_deg = 0
    if len(calibration_points) >= 2:
        p1, p2 = calibration_points[-2], calibration_points[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        # 修正旋转方向：顺时针为负，逆时针为正
        auto_rotate_deg = angle_deg - 90 
        print(f"自动矫正角度: {auto_rotate_deg:.2f}°")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2. 初始化并行执行环境
    frames_data = [] # 数据存储
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_heavy.task'))
    
    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer(frames_data=frames_data)
    else:
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm, frames_data=frames_data)

    window_name = 'NintAi Parallel Processing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, int(640 * height / width))

    fvs = FileVideoStream(args.input).start()
    
    # 重要性能优化：使用 initializer 缓存模型实例
    num_workers = multiprocessing.cpu_count() - 1
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_path, args.gpu)
    )
    
    futures = []
    results_cache = {}
    next_frame_idx = 0
    submitted_count = 0

    print(f"并行引擎启动 (核心数: {num_workers})...")

    try:
        while fvs.more() or futures:
            while fvs.more() and len(futures) < num_workers * 2:
                frame = fvs.read()
                # 修复：只传递 3 个参数 (frame, frame_idx, rotate_deg)
                futures.append(executor.submit(
                process_frame_worker, frame, submitted_count, auto_rotate_deg))
                submitted_count += 1

            done_list = [f for f in futures if f.done()]
            for f in done_list:
                f_idx, p_frame, res = f.result()
                results_cache[f_idx] = (p_frame, res)
                futures.remove(f)

            while next_frame_idx in results_cache:
                p_frame, res = results_cache.pop(next_frame_idx)
                processed_frame, _ = analyzer.process(p_frame, res, next_frame_idx)
                cv2.imshow(window_name, processed_frame)
                next_frame_idx += 1

            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt:
        print("\n用户中断处理")
    finally:
        fvs.stop()
        executor.shutdown(wait=False)
        cv2.destroyAllWindows()

    # 3. 报告生成 (移除重复逻辑)
    if not frames_data:
        print("未检测到有效姿态数据")
        sys.exit()

    if args.view == 'front':
        df_front = pd.DataFrame([
            {
                'frame_idx': f['frame_idx'],
                'left_knee_x_offset': f['angles'].get('left_knee_x_offset', 0),
                'right_knee_x_offset': f['angles'].get('right_knee_x_offset', 0)
            } for f in frames_data
        ])
        
        stats = {
            'left_valgus_std': df_front['left_knee_x_offset'].std(),
            'right_valgus_std': df_front['right_knee_x_offset'].std(),
            'left_valgus_avg': df_front['left_knee_x_offset'].mean(),
            'right_valgus_avg': df_front['right_knee_x_offset'].mean()
        }
        
        prompt = ai_report.generate_diagnostic_prompt(stats, 'front')
        print("\n" + "="*20 + " 诊断结论 " + "="*20)
        print(prompt)

        report_path = os.path.join(report_dir, "front_view_diagnosis.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"\n[系统提示] 报告已保存至: {report_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()