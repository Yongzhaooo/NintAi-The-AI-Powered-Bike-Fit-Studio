'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\NintAi-The-AI-Powered-Bike-Fit-Studio\\nvcheck.py
Author       : Yongzhao Chen
Date         : 2026-03-06 02:13:13
LastEditTime : 2026-03-06 02:13:13
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
'''
import onnxruntime as ort

# 获取所有可用的推理后端
providers = ort.get_available_providers()
print(f"所有可用后端: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ 状态：CUDA 后端已识别。")
    try:
        # 尝试创建一个测试会话来加载 DLL
        sess = ort.InferenceSession(None, providers=['CUDAExecutionProvider'])
        print("🚀 验证通过：DLL 加载正常，GPU 已准备就绪！")
    except Exception as e:
        if "failed to create CUDAExecutionProvider" in str(e).lower():
            print(f"❌ 报错：虽然看到了 CUDA，但 DLL 加载失败。原因通常是 cuDNN 没放对地方。")
        else:
            # 这里的报错如果是空的或者关于模型路径的，通常说明后端本身已经通了
            print(f"✅ 验证通过：后端引擎初始化成功。")
else:
    print("❌ 状态：依然只有 CPU。请检查 CUDA 12 的 bin 目录是否在环境变量 PATH 中。")