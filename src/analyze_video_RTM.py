# src/analyze_video_RTM.py
import cv2
import argparse
import sys
import os
import math
import time
import multiprocessing as mp
from queue import Empty

# --- 环境兼容 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src import tracking, analyzers

# 全局变量用于鼠标回调
calibration_points = []

def select_vertical_line(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(param, (x, y), 10, (0, 0, 255), -1)
        if len(calibration_points) >= 2:
            cv2.line(param, calibration_points[-2], calibration_points[-1], (0, 255, 0), 3)
        cv2.imshow('Calibration', param)

def pose_worker_rtm(input_queue, output_queue, mode):
    import os
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if os.path.exists(cuda_path):
        os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']
        try: os.add_dll_directory(cuda_path)
        except: pass
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
    
    from src.tracking import PoseDetectorRTM
    detector = PoseDetectorRTM(device='cuda', mode=mode)
    
    while True:
        try:
            task = input_queue.get(timeout=10)
            if task is None: break
            f_idx, proxy_frame, scale = task
            results = detector.predict_image(proxy_frame)
            lm_dict = detector.get_landmarks_dict(results, proxy_frame.shape)
            for k in lm_dict:
                lm_dict[k][0] *= scale
                lm_dict[k][1] *= scale
            output_queue.put((f_idx, lm_dict))
        except: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--view", "-v", type=str, choices=['side', 'front', 'back'], default='side')
    parser.add_argument("--side", type=str, choices=['left', 'right', 'auto'], default='auto')
    parser.add_argument("--crank_mm", type=float, default=170.0)
    parser.add_argument("--mode", type=str, default='lightweight')
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--rotate", type=float, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -- determine display window size (max 1/4 screen area) --
    display_w = args.width
    display_h = int(display_w * orig_h / orig_w)
    if not args.headless:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = display_w * 2, display_h * 2
        max_w, max_h = screen_w // 2, screen_h // 2   # half each side → 1/4 area
        display_w = min(display_w, max_w)
        display_h = int(display_w * orig_h / orig_w)
        if display_h > max_h:
            display_h = max_h
            display_w = int(display_h * orig_w / orig_h)

    # 1. 交互式校准逻辑 (非 headless 且边视或正面皆可)
    auto_rotate_deg = args.rotate
    if not args.headless and args.view in ('side', 'front'):
        ret, first_frame = cap.read()
        if ret:
            cal_img = cv2.resize(first_frame, (display_w, display_h))
            cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
            cv2.imshow('Calibration', cal_img)
            cv2.setMouseCallback('Calibration', select_vertical_line, cal_img)
            print("👉 [交互校准] 请点击两点定义垂线。完成后按 'Space' 或 'Enter' 继续...")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in [13, 32]: break
            if len(calibration_points) >= 2:
                p1, p2 = calibration_points[-2], calibration_points[-1]
                auto_rotate_deg = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) - 90
            cv2.destroyWindow('Calibration')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2. 启动 GPU 进程（proxy 分辨率可根据视角降低以提升侧面速度）
    proxy_w = 640 if args.view == 'front' else 480
    scale = orig_w / proxy_w
    proxy_h = int(orig_h / scale)
    input_q, output_q = mp.Queue(maxsize=30), mp.Queue()
    worker = mp.Process(target=pose_worker_rtm, args=(input_q, output_q, args.mode))
    worker.start()

    # 3. 根据参数初始化分析器
    if args.view == 'front':
        analyzer = analyzers.FrontViewAnalyzer(crank_mm=args.crank_mm)
    elif args.view == 'back':
        analyzer = analyzers.BackViewAnalyzer()
    else:
        analyzer = analyzers.SideViewAnalyzer(crank_mm=args.crank_mm, target_side=None if args.side == 'auto' else args.side)

    raw_cache, res_cache = {}, {}
    sub_idx, proc_idx = 0, 0
    start_t = time.time()

    try:
        while proc_idx < sub_idx or cap.isOpened():
            if cap.isOpened() and input_q.qsize() < 20:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    input_q.put(None)
                else:
                    if auto_rotate_deg != 0:
                        M = cv2.getRotationMatrix2D((orig_w//2, orig_h//2), auto_rotate_deg, 1.0)
                        frame = cv2.warpAffine(frame, M, (orig_w, orig_h))
                    proxy = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_NEAREST)
                    raw_cache[sub_idx] = frame
                    input_q.put((sub_idx, proxy, scale))
                    sub_idx += 1

            while not output_q.empty():
                f_idx, lm_dict = output_q.get_nowait()
                res_cache[f_idx] = lm_dict

            while proc_idx in res_cache:
                lm_dict = res_cache.pop(proc_idx)
                f = raw_cache.pop(proc_idx)
                proc_frame, _ = analyzer.process(f, lm_dict, proc_idx)

                if not args.headless:
                    show = cv2.resize(proc_frame, (display_w, display_h))
                    cv2.imshow("NintAi Turbo", show)
                    if cv2.waitKey(1) == ord('q'): 
                        cap.release(); break
                
                proc_idx += 1
                if proc_idx % 30 == 0:
                    fps = proc_idx/(time.time()-start_t)
                    print(f"\r🔥 FPS: {fps:.1f} | 进度: {proc_idx}/{total_f}", end="")
    except KeyboardInterrupt:
        print("\n🛑 处理中断（Ctrl-C）")
    finally:
        worker.terminate(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    # required on Windows to avoid recursive process spawning
    mp.freeze_support()
    main()