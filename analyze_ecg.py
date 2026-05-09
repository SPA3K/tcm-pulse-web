#!/usr/bin/env python3
"""
岐黄脉镜 - ECG 分析脚本
分析 Apple Health 导出的心电图数据
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path

def parse_ecg_csv(filepath):
    """解析 Apple Health ECG CSV 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取元数据
    metadata = {}
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('姓名,'):
            metadata['name'] = line.split(',')[1].strip()
        elif line.startswith('出生日期,'):
            metadata['birth_date'] = line.split(',')[1].strip().replace('"', '')
        elif line.startswith('记录日期,'):
            metadata['record_date'] = line.split(',')[1].strip()
        elif line.startswith('分类,'):
            metadata['classification'] = line.split(',')[1].strip()
        elif line.startswith('症状,'):
            metadata['symptoms'] = line.split(',')[1].strip() or '无'
        elif line.startswith('采样速率,'):
            metadata['sampling_rate'] = int(line.split(',')[1].strip().replace('赫兹', ''))
        elif line.startswith('单位,'):
            metadata['unit'] = line.split(',')[1].strip()
            data_start = i + 2
            break

    # 提取 ECG 数据
    ecg_values = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            try:
                ecg_values.append(float(line))
            except ValueError:
                continue

    metadata['ecg_signal'] = np.array(ecg_values)
    metadata['duration'] = len(ecg_values) / metadata['sampling_rate']

    return metadata

def extract_features(ecg_data):
    """提取 18 维特征"""
    signal = ecg_data['ecg_signal']
    sampling_rate = ecg_data['sampling_rate']

    features = {}

    # 1. ECG 信号处理
    try:
        # 清洗信号
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)

        # 检测 R 峰
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        rpeaks_indices = rpeaks['ECG_R_Peaks']

        if len(rpeaks_indices) < 3:
            print("⚠️  检测到的心跳过少，无法进行完整分析")
            return None

        # 计算 RR 间期（毫秒）
        rr_intervals = np.diff(rpeaks_indices) / sampling_rate * 1000

        # 2. 频率维度（3维）
        hr = 60000 / rr_intervals  # 心率 BPM
        features['mean_hr'] = np.mean(hr)
        features['max_hr'] = np.max(hr)
        features['min_hr'] = np.min(hr)

        # 3. 节律维度（4维）
        features['rr_cv'] = np.std(rr_intervals) / np.mean(rr_intervals)  # RR 变异系数

        # 检测早搏（RR 间期突然缩短）
        rr_diff = np.diff(rr_intervals)
        features['premature_beats'] = np.sum(rr_diff < -100)  # 提前超过100ms

        # 检测停搏（RR 间期突然延长）
        features['pauses'] = np.sum(rr_diff > 300)  # 延长超过300ms

        # 检测规律性停搏
        features['regular_pause'] = 0  # 简化处理

        # 4. 振幅维度（3维）
        r_amplitudes = signal[rpeaks_indices]
        features['r_amplitude_mean'] = np.mean(r_amplitudes)
        features['r_amplitude_std'] = np.std(r_amplitudes)
        features['r_amplitude_cv'] = features['r_amplitude_std'] / abs(features['r_amplitude_mean']) if features['r_amplitude_mean'] != 0 else 0

        # 5. 张力维度（4维 - HRV 分析）
        features['sdnn'] = np.std(rr_intervals)  # 时域标准差

        # RMSSD (连续RR间期差值的均方根)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        features['rmssd'] = rmssd

        # pNN50 (相邻RR间期差值>50ms的百分比)
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        features['pnn50'] = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0

        # LF/HF 比值（简化处理）
        features['lf_hf_ratio'] = 2.0  # 默认值，完整版需要频域分析

        # 6. 流畅度维度（2维）
        # Poincaré SD1
        features['sd1'] = np.std(np.diff(rr_intervals) / np.sqrt(2))

        # RR 粗糙度
        features['rr_roughness'] = np.mean(np.abs(np.diff(rr_intervals)))

        # 7. 波形维度（2维）
        # QRS 宽度（简化）
        features['qrs_width'] = 80  # 默认值（ms）

        # 信号质量
        features['signal_quality'] = 0.9  # 简化评分

        return features, rr_intervals, hr

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return None

def classify_pulse(features):
    """脉象分类"""
    candidates = []

    mean_hr = features['mean_hr']
    rmssd = features['rmssd']
    rr_cv = features['rr_cv']
    r_amp = features['r_amplitude_mean']

    # 1. 迟脉 (HR < 60)
    if mean_hr < 60:
        confidence = min(0.95, 0.7 + (60 - mean_hr) / 100)
        candidates.append({
            'type': '迟脉',
            'confidence': confidence,
            'evidence': f'平均心率 {mean_hr:.1f} bpm < 60',
            'meaning': '寒证、阳虚'
        })

    # 2. 数脉 (HR > 90)
    elif mean_hr > 90:
        confidence = min(0.95, 0.7 + (mean_hr - 90) / 100)
        candidates.append({
            'type': '数脉',
            'confidence': confidence,
            'evidence': f'平均心率 {mean_hr:.1f} bpm > 90',
            'meaning': '热证、阴虚'
        })

    # 3. 疾脉 (HR > 120)
    if mean_hr > 120:
        candidates.append({
            'type': '疾脉',
            'confidence': 0.90,
            'evidence': f'平均心率 {mean_hr:.1f} bpm > 120',
            'meaning': '阳极阴竭'
        })

    # 4. 弦脉 (RMSSD < 20 + LF/HF > 2.0)
    if rmssd < 20:
        candidates.append({
            'type': '弦脉',
            'confidence': 0.65,
            'evidence': f'RMSSD {rmssd:.1f}ms < 20，张力增高',
            'meaning': '肝胆病、痛证'
        })

    # 5. 缓脉 (60 <= HR <= 75, RMSSD > 50)
    if 60 <= mean_hr <= 75 and rmssd > 50:
        candidates.append({
            'type': '缓脉',
            'confidence': 0.60,
            'evidence': f'心率 {mean_hr:.1f} bpm 适中，RMSSD {rmssd:.1f}ms 良好',
            'meaning': '平脉或脾虚'
        })

    # 6. 滑脉 (RR变异系数 < 0.03)
    if rr_cv < 0.03:
        candidates.append({
            'type': '滑脉',
            'confidence': 0.50,
            'evidence': f'RR变异系数 {rr_cv:.3f} < 0.03，节律流畅',
            'meaning': '痰湿、妊娠'
        })

    # 7. 促脉/结脉 (有早搏或停搏)
    if features['premature_beats'] > 0:
        if mean_hr > 90:
            candidates.append({
                'type': '促脉',
                'confidence': 0.85,
                'evidence': f'快心率 + 早搏 {features["premature_beats"]} 次',
                'meaning': '阳盛实热、瘀血'
            })
        else:
            candidates.append({
                'type': '结脉',
                'confidence': 0.85,
                'evidence': f'慢心率 + 早搏 {features["premature_beats"]} 次',
                'meaning': '阴盛气结、寒痰'
            })

    # 8. 洪脉 (振幅大)
    if r_amp > 100:  # µV
        candidates.append({
            'type': '洪脉',
            'confidence': 0.55,
            'evidence': f'R波振幅 {r_amp:.1f} µV，波形宽大',
            'meaning': '气分热盛'
        })

    # 9. 细脉 (振幅小)
    elif abs(r_amp) < 50:
        candidates.append({
            'type': '细脉',
            'confidence': 0.55,
            'evidence': f'R波振幅 {abs(r_amp):.1f} µV，波形细小',
            'meaning': '气血两虚'
        })

    # 如果没有检测到异常，判定为平脉
    if not candidates:
        candidates.append({
            'type': '平脉',
            'confidence': 0.80,
            'evidence': '各项指标均在正常范围',
            'meaning': '正常脉象，有胃气'
        })

    # 按置信度排序
    candidates.sort(key=lambda x: x['confidence'], reverse=True)

    return candidates

def generate_report(metadata, features, pulse_types):
    """生成诊断报告"""
    print("\n" + "="*60)
    print("岐黄脉镜 - 中医脉诊分析报告")
    print("="*60)

    print(f"\n【基本信息】")
    print(f"  姓名: {metadata['name']}")
    print(f"  记录时间: {metadata['record_date']}")
    print(f"  Apple 分类: {metadata['classification']}")
    print(f"  症状: {metadata['symptoms']}")
    print(f"  数据时长: {metadata['duration']:.1f} 秒")

    print(f"\n【18维特征提取】")
    print(f"  ├─ 频率维度:")
    print(f"  │   平均心率: {features['mean_hr']:.1f} bpm")
    print(f"  │   最大心率: {features['max_hr']:.1f} bpm")
    print(f"  │   最小心率: {features['min_hr']:.1f} bpm")
    print(f"  ├─ 节律维度:")
    print(f"  │   RR变异系数: {features['rr_cv']:.3f}")
    print(f"  │   早搏次数: {features['premature_beats']}")
    print(f"  │   停搏次数: {features['pauses']}")
    print(f"  ├─ 振幅维度:")
    print(f"  │   R波平均振幅: {features['r_amplitude_mean']:.1f} µV")
    print(f"  │   R波振幅标准差: {features['r_amplitude_std']:.1f} µV")
    print(f"  ├─ 张力维度 (HRV):")
    print(f"  │   SDNN: {features['sdnn']:.1f} ms")
    print(f"  │   RMSSD: {features['rmssd']:.1f} ms")
    print(f"  │   pNN50: {features['pnn50']:.1f} %")
    print(f"  └─ 流畅度维度:")
    print(f"      Poincaré SD1: {features['sd1']:.1f} ms")
    print(f"      RR粗糙度: {features['rr_roughness']:.1f} ms")

    print(f"\n【脉象诊断】")
    for i, pulse in enumerate(pulse_types[:3], 1):
        if i == 1:
            print(f"  【主脉】{pulse['type']} (置信度: {pulse['confidence']*100:.0f}%)")
        else:
            print(f"  【兼脉】{pulse['type']} (置信度: {pulse['confidence']*100:.0f}%)")
        print(f"    证据: {pulse['evidence']}")
        print(f"    临床意义: {pulse['meaning']}")
        print()

    print("="*60)
    print("⚠️  本分析仅供参考，不能替代专业中医师诊断")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 分析最新的心电图记录
    ecg_file = Path("/tmp/apple_health_export/electrocardiograms/ecg_2026-05-09.csv")

    print("📊 正在解析 ECG 数据...")
    metadata = parse_ecg_csv(ecg_file)

    print(f"✅ 数据加载完成: {len(metadata['ecg_signal'])} 个采样点")
    print(f"   采样率: {metadata['sampling_rate']} Hz")
    print(f"   时长: {metadata['duration']:.1f} 秒\n")

    print("🔬 正在提取 18 维特征...")
    result = extract_features(metadata)

    if result is None:
        print("❌ 特征提取失败")
        exit(1)

    features, rr_intervals, hr = result
    print(f"✅ 特征提取完成\n")

    print("🏥 正在进行脉象分类...")
    pulse_types = classify_pulse(features)
    print(f"✅ 识别到 {len(pulse_types)} 种脉象特征\n")

    # 生成报告
    generate_report(metadata, features, pulse_types)
