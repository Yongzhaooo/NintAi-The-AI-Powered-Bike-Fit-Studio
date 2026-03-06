# src/ai_report.py



def generate_front_prompt(stats):
    return f"""
# 🚴 NintAi 骑行生物力学诊断申请书 (正面视角)
**核心数据 (横向追踪)**:
* **右腿膝盖横向偏移均值**: {stats.get('right_valgus_avg', 0):.2f} px
* **右腿偏移标准差 (稳定性)**: {stats.get('right_valgus_std', 0):.2f} (重点关注)
* **左腿膝盖横向偏移均值**: {stats.get('left_valgus_avg', 0):.2f} px
* **左腿偏移标准差 (稳定性)**: {stats.get('left_valgus_std', 0):.2f}

**诊断请求**:
你现在是一名顶级的 Fitter。请根据以上数据分析我的**膝盖追踪 (Knee Tracking)** 问题。
... (其余部分保持不变)
"""

def generate_side_prompt(stats):
    return f"""
# 🚴 NintAi 骑行生物力学诊断申请书 (侧面视角)
**1. 核心角度数据**:
| 指标 | 测量值 | 参考范围 | 状态 |
| :--- | :--- | :--- | :--- |
| **最大膝盖伸展 (Knee Ext)** | {stats.get('knee_ext_max', 0):.1f}° | 140° - 150° | - |
| **最小膝盖压缩 (Knee Flex)** | {stats.get('knee_flex_min', 0):.1f}° | 70° - 75° | - |
| **最小髋关节闭合 (Hip Closed)** | {stats.get('hip_closed_min', 0):.1f}° | > 45° | - |
| **平均躯干倾角 (Torso)** | {stats.get('back_avg', 0):.1f}° | 40° - 50° | - |
| **平均脚踝角度 (Foot)** | {stats.get('foot_angle_avg', 0):.1f}° | 0° - 20° | - |
...
"""

def generate_diagnostic_prompt(stats, view_type):
    """
    统一入口函数，根据视角调用对应的 Prompt 生成逻辑
    """
    if view_type == 'front':
        return generate_front_prompt(stats)
    elif view_type == 'side':
        return generate_side_prompt(stats)
    return "未知的视角类型"