import cv2
import argparse
import sys
import os
import math
import time
import multiprocessing as mp
from queue import Empty

# --- 路径兼容 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src import tracking, analyzers

def pose_worker_rtm(input_queue, output_queue, mode):
    import os
    # 强制 DLL 注入
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if os.path.exists(cuda_path):
        os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']
        try: os.add_dll_directory(cuda_path)
        except: pass

    # 显式禁用 TensorRT 引擎缓存以防首次启动卡死
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
            # 坐标还原到原始比例
            for k in lm_dict:
                lm_dict[k][0] *= scale
                lm_dict[k][1] *= scale
            output_queue.put((f_idx, lm_dict))
        except: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--mode", type=str, default='lightweight')
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 代理帧参数：减少传输体积的核心
    proxy_w = 640
    scale = orig_w / proxy_w
    proxy_h = int(orig_h / scale)

    # 启动 GPU 进程
    input_q, output_q = mp.Queue(maxsize=30), mp.Queue()
    worker = mp.Process(target=pose_worker_rtm, args=(input_q, output_q, args.mode))
    worker.start()

    analyzer = analyzers.SideViewAnalyzer(crank_mm=170.0)
    raw_cache, res_cache = {}, {}
    sub_idx, proc_idx = 0, 0
    start_t = time.time()

    print(f"🚀 开始分析... 推理代理尺寸: {proxy_w}x{proxy_h}")

    try:
        while proc_idx < sub_idx or cap.isOpened():
            # 生产者
            if cap.isOpened() and input_q.qsize() < 20:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    input_q.put(None)
                else:
                    proxy = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_NEAREST)
                    raw_cache[sub_idx] = frame
                    input_q.put((sub_idx, proxy, scale))
                    sub_idx += 1

            # 消费者
            while not output_q.empty():
                f_idx, lm_dict = output_q.get_nowait()
                res_cache[f_idx] = lm_dict

            while proc_idx in res_cache:
                lm_dict = res_cache.pop(proc_idx)
                f = raw_cache.pop(proc_idx)
                proc_frame, _ = analyzer.process(f, lm_dict, proc_idx)

                if not args.headless:
                    show = cv2.resize(proc_frame, (args.width, int(args.width*orig_h/orig_w)))
                    cv2.imshow("NintAi Turbo", show)
                    if cv2.waitKey(1) == ord('q'): 
                        cap.release(); break
                
                proc_idx += 1
                if proc_idx % 30 == 0:
                    print(f"\r🔥 速度: {proc_idx/(time.time()-start_t):.1f} FPS | 进度: {proc_idx}/{total_f}", end="")

    finally:
        worker.terminate(); cap.release(); cv2.destroyAllWindows()
        print(f"\n任务完成。平均速度: {proc_idx/(time.time()-start_t):.1f} FPS")

if __name__ == "__main__":
    mp.freeze_support()
    main()