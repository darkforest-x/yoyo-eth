# ITERATION V1 — Correctness Repair + Causal Anchor

状态:**completed**。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_iteration_v1.py --config configs/iteration_v1.yaml --force
python3 -m pytest tests/ -q
```

## Setup

| item | value |
|---|---|
| data | 2025-06-01 12:45:00+00:00 .. 2026-05-03 23:45:00+00:00, 32301 bars, gaps=0 |
| boundary | 2026-05-04T00:00:00+00:00(边界后行在 CSV 解析后、任何指标/扫描/特征/标签/训练之前剔除)|
| threshold | 1.8091 = q0.3 on train bars only |
| split | train_end=22610 (2026-01-23 01:00:00+00:00), val_end=27455 (2026-03-14 12:15:00+00:00), gap=164 |
| model | LightGBM fixed params, subsample=0.8 subsample_freq=1 seed=42 |
| manifest | git=cd7a311a dirty=True csv_sha=9cca1f7f5dbb.. |

## 正确性指标(doc 9.1)

| item | value |
|---|---|
| Legacy raw candidates | 8737 |
| Legacy dedup events | 321 |
| Compression episodes | 330 |
| Episodes with trigger | 233 |
| Episodes without trigger | 97 |
| Anchored events | 233 |
| Legacy splits | {'train': 229, 'val': 44, 'test': 40} |
| Anchored splits | {'train': 171, 'val': 31, 'test': 27} |

## 语义对齐(doc 9.2)

| quantity | arm | mean | median | p25 | p75 |
|---|---|---|---|---|---|
| compression_duration_at_decision | legacy | +3.000 | +3.000 | +3.000 | +3.000 |
| compression_duration_at_decision | anchored | +8.930 | +6.000 | +4.000 | +11.000 |
| episode_age_at_decision | legacy | +2.000 | +2.000 | +2.000 | +2.000 |
| episode_age_at_decision | anchored | +8.013 | +5.000 | +3.000 | +10.000 |
| failed_reclaim_count_10 | legacy | +3.885 | +4.000 | +2.000 | +6.000 |
| failed_reclaim_count_10 | anchored | +4.210 | +4.000 | +2.000 | +6.000 |
| cluster_penetration_count_10 | legacy | +1.693 | +1.000 | +1.000 | +3.000 |
| cluster_penetration_count_10 | anchored | +1.847 | +1.000 | +1.000 | +3.000 |
| ema20_slope_5 | legacy | -0.064 | -0.024 | -0.506 | +0.384 |
| ema20_slope_5 | anchored | -0.398 | -0.320 | -0.575 | -0.128 |
| close_to_cluster_atr | legacy | -0.148 | -0.104 | -1.146 | +0.838 |
| close_to_cluster_atr | anchored | -1.451 | -1.169 | -1.848 | -0.783 |
| ma_dispersion_slope_5 | legacy | -0.638 | -0.575 | -0.838 | -0.357 |
| ma_dispersion_slope_5 | anchored | -0.338 | -0.336 | -0.580 | -0.083 |

按 trigger_type(anchored,val+test+train 全体):

| trigger_type | n | mean utility | mean net |
|---|---|---|---|
| compression_exit_down | 12 | +0.429 | -29.7bp |
| compression_exit_down+failed_reclaim | 1 | -1.493 | -173.4bp |
| compression_exit_down+local_low_break | 1 | +3.442 | +123.7bp |
| failed_reclaim | 98 | +1.258 | -8.4bp |
| failed_reclaim+local_low_break | 52 | +0.664 | -21.8bp |
| local_low_break | 65 | +1.218 | +22.0bp |

## Legacy 模型结果

| split | n | pearson | spearman | group | n | mean util | median util | mean MFE | mean MAE | gross | net@0.001 | net@0.002 | pos-net@0.001 | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 44 | +0.084 | +0.008 | top_10pct | 5 | +0.198 | +0.104 | 1.31 | 1.58 | -14.9bp | -24.9bp | -34.9bp | 0.40 | LOW_SAMPLE |
| validation | 44 | +0.084 | +0.008 | top_20pct | 9 | +0.032 | +0.104 | 1.50 | 2.10 | -22.8bp | -32.8bp | -42.8bp | 0.33 | LOW_SAMPLE |
| validation | 44 | +0.084 | +0.008 | all | 44 | +0.711 | +0.441 | 2.57 | 2.66 | -14.6bp | -24.6bp | -34.6bp | 0.43 |  |
| validation | 44 | | | matched control | 880 | +1.105 | +0.788 | 2.95 | 2.63 | +18.2bp | +8.2bp | -1.8bp | 0.52 | req=880 uniq=787 |
| test | 40 | -0.035 | +0.000 | top_10pct | 4 | +2.099 | +0.081 | 3.69 | 2.28 | -15.3bp | -25.3bp | -35.3bp | 0.50 | LOW_SAMPLE |
| test | 40 | -0.035 | +0.000 | top_20pct | 8 | -0.488 | -0.418 | 2.34 | 4.04 | -87.7bp | -97.7bp | -107.7bp | 0.38 | LOW_SAMPLE |
| test | 40 | -0.035 | +0.000 | all | 40 | +0.019 | -0.145 | 2.48 | 3.51 | -39.1bp | -49.1bp | -59.1bp | 0.35 |  |
| test | 40 | | | matched control | 800 | +0.816 | +0.494 | 2.81 | 2.84 | -2.6bp | -12.6bp | -22.6bp | 0.45 | req=800 uniq=720 |

## Anchored 模型结果

| split | n | pearson | spearman | group | n | mean util | median util | mean MFE | mean MAE | gross | net@0.001 | net@0.002 | pos-net@0.001 | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 31 | +0.374 | +0.465 | top_10pct | 4 | +0.461 | +0.292 | 1.54 | 1.54 | -35.3bp | -45.3bp | -55.3bp | 0.25 | LOW_SAMPLE |
| validation | 31 | +0.374 | +0.465 | top_20pct | 7 | +1.526 | +0.481 | 2.54 | 1.44 | +17.1bp | +7.1bp | -2.9bp | 0.43 | LOW_SAMPLE |
| validation | 31 | +0.374 | +0.465 | all | 31 | +0.296 | +0.090 | 2.38 | 2.98 | -27.9bp | -37.9bp | -47.9bp | 0.29 |  |
| validation | 31 | | | matched control | 620 | +1.131 | +0.939 | 2.99 | 2.66 | +14.1bp | +4.1bp | -5.9bp | 0.54 | req=620 uniq=581 |
| test | 27 | -0.074 | -0.218 | top_10pct | 3 | -0.027 | +1.472 | 3.07 | 4.42 | -111.0bp | -121.0bp | -131.0bp | 0.33 | LOW_SAMPLE |
| test | 27 | -0.074 | -0.218 | top_20pct | 6 | -0.456 | -0.412 | 2.27 | 3.89 | -129.8bp | -139.8bp | -149.8bp | 0.17 | LOW_SAMPLE |
| test | 27 | -0.074 | -0.218 | all | 27 | +2.044 | +0.419 | 3.94 | 2.71 | -8.7bp | -18.7bp | -28.7bp | 0.33 |  |
| test | 27 | | | matched control | 540 | +0.870 | +0.636 | 2.89 | 2.89 | -3.3bp | -13.3bp | -23.3bp | 0.45 | req=540 uniq=504 |

<!-- Analyst narrative (15 questions, FACT/OBSERVATION/HYPOTHESIS/DECISION) appended below. -->

## 15 问逐条回答

1. **随机对照重复 merge bug 是否确认存在?** FACT:存在。`matched_random_control` 用
   `replace=True` 抽样后,旧 `add_labels` 按可重复的 `decision_pos` merge,k 次重复
   膨胀为 k×k 行。MVP 报告的 1076(应 880)/978(应 800)即该 bug 的直接证据。
2. **修复前后 control sample count?** FACT:修复前 validation 1076 / test 978(MVP
   报告存档);修复后 validation 880 / test 800,由运行时断言强制(不等即整轮失败)。
3. **修复后 validation 是否严格 880?** FACT:是(requested=actual=880,unique=787,
   重复抽样 93 次为合法 bootstrap 重复,各带独立权重)。
4. **修复后 test 是否严格 800?** FACT:是(800/800,unique=720)。
5. **Legacy decision bar 平均在 episode 什么位置?** FACT:恒在 episode 第 3 根
   (qualification 那根,episode_age=2,compression_duration=3,全体常数)。episode
   中位总长 22 根 → legacy 永远在压缩带**开头 ~14%** 处决策。
6. **Anchored 比 Legacy 晚多少根?** FACT:平均晚 **6.0 根**(episode_age 均值 8.0 vs
   2.0;中位 5 vs 2)。
7. **三种 trigger 数量与占比?** FACT(入数据集的 229 事件):failed_reclaim 98
   (42.8%)、local_low_break 65(28.4%)、failed_reclaim+local_low_break 52
   (22.7%)、compression_exit_down 12(5.2%)、双组合各 1(0.9%)。
   compression_exit_down 稀少:带内价格通常已先触发 A/C。
8. **语义特征分布是否更合理?** FACT:`compression_duration_at_decision` 从常数 3 变为
   分布(median 6, p75 ~11);`close_to_cluster_atr` −0.15→−1.45(决策时价格已破位到
   簇下方 1.5 ATR);`ema20_slope_5` −0.06→−0.40(短均线明确下弯,部分由触发条件
   构造保证)。OBSERVATION:`failed_reclaim_count_10`/`cluster_penetration_count_10`
   两臂差异不大(3.9→4.2 / 1.7→1.8)——回抽失败的"密度"没变,变的是决策时点相对
   破位的位置。
9. **Legacy 排序结果?** FACT:val ρ=+0.008(p=0.96)、test ρ=+0.000(p=1.00)——
   本轮启用 subsample_freq=1 后 Legacy 彻底无排序能力(MVP 基线的 +0.166/−0.068
   本就在噪声内)。Top10 全部 LOW_SAMPLE。
10. **Anchored 排序结果?** FACT:val ρ=+0.465(p=0.01, n=31)、test ρ=−0.218
    (p=0.27, n=27)。OBSERVATION:val 显著为正但 test 翻负;n=31/27 的噪声带约
    ±0.36/±0.39,不构成稳定排序能力的证据。HYPOTHESIS:val 段(2026-01→02,下行)
    与 anchored 触发语义同向,test 段(2026-02→05,横盘转震荡)不同向——regime
    依赖,需 walk-forward 复检。
11. **两臂相对 matched control?** FACT:Legacy all−ctrl:val −32.8bp / test −36.5bp;
    Anchored all−ctrl:val −42.0bp / test −5.4bp。两臂池级都不跑赢同月同波动桶随机
    入场;MVP 报告中 −34.5/−37.1bp 的旧值作废,以本轮修复后数字为准(方向结论
    不变)。
12. **Anchored 是否改善人工视觉语义一致性?** OBSERVATION:决策图(盲审版,无任何
    未来字段)抽样显示 anchored 决策点落在"密集平台边缘破位/回抽失败"处,与目标
    形态语义显著更接近(legacy 落在带内早期,盘面尚无方向)。最终裁决需 owner 翻阅
    `reports/iteration_v1/review_charts/`(legacy/anchored × validation/test,各
    top/random/bottom ≤20 张,decision 与 outcome 同名对应)。
13. **当前结果是否具备统计力?** FACT:不具备。anchored val/test = 31/27,Top10 组
    3–4 个样本(全 LOW_SAMPLE 标记),分 trigger × split 后每格 1–12 个样本。本轮
    回答的是语义对齐与正确性,不是收益判断。
14. **最大剩余问题?** anchored 事件量(233,进数据集 229)在单一 70/15/15 切分下
    val/test 太薄;且 val(下行)与 test(横盘)regime 不同,任何排序结论都被切分
    位置绑架。次要:6h 固定 horizon 对"破位后行情"可能过短/过长,未验证。
15. **下一轮建议?** DECISION(建议,owner 定):优先**不改 Anchor、不改标签**,把
    anchored 管道接入已有的 4 折 walk-forward harness(P02 基建现成,OOS 可到
    ~200 事件),复检 val 上 ρ=+0.465 是否幸存;若幸存,再谈标签/路径工作。若不
    幸存,该语义在 6h short_utility 下按停止规则处理。

## 风险与诚实声明

- 基线偏移:任务书基线 55ddf5d,实际基于 cd7a311(P02 已并入;未覆盖任何 MVP 产物)。
- P02 的 per-fold 对照组数字带同一 fanout bug(每折事件数 × 20 vs 实报偏大),
  本轮修复后函数已正确,但 P02 未重跑;其 all-vs-ctrl(−6.7~+5.7bp)量级应视为
  近似。P02 的池化 spearman 结论不受影响(不依赖对照组)。
- 本轮启用 subsample_freq=1(修复 5.4)使模型与 MVP/P02 的模型不逐位可比;两臂
  之间严格可比(同参数同种子)。
- anchored val ρ=+0.465 (p=0.01) 是 n=31 的单切分结果,且本轮同时看了 val 和 test
  (test 翻负)——按纪律只记 OBSERVATION,不作 edge 声明。
- trigger 条件含 `ema20_slope_5 < 0`,语义对齐指标中该特征的两臂差异部分是构造使然,
  已在第 8 问注明。
- holdout(≥2026-05-04)0 消耗;数据在 CSV 解析后立即剔除边界后行——是代码级排除,
  非物理隔离。
- 第三轮独立对抗审查:无 CRITICAL/HIGH;4 个低危健壮性问题(空触发崩溃、error-dict
  防护、成本列硬编码、测试场景构造)已修复,数字逐位不变。

(iteration_v1 到此停止:未改标签、未扫超参、未加特征、未动 holdout、未接执行层。)
