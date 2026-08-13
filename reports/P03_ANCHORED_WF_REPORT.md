# P03 — Anchored 触发的 walk-forward 复检

问题:iteration_v1 的 anchored val ρ=+0.465(p=0.01, n=31)是真信号还是单切分
regime 巧合?方法:与 P02 完全相同的 4 折 anchored harness,两臂(legacy
zone_start 对照 / anchored 因果触发)同参数(iteration_v1 模型配置,
subsample_freq=1)、同阈值冻结纪律、同修复后对照组。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_anchored_walkforward.py
python3 -m pytest tests/ -q
```

| arm | OOS n | rho per fold | rho wmean | top10 gross | all gross | control gross | all-ctrl |
|---|---|---|---|---|---|---|---|
| legacy_zone_start | 195 | [-0.129, 0.029, 0.013, -0.249] | -0.075 | +32.5bp | +14.1bp | +9.3bp | +4.8bp |
| anchored | 140 | [0.085, 0.064, 0.219, -0.151] | +0.065 | +35.7bp | +3.0bp | +8.4bp | -5.4bp |

folds: [(12920, 12920, 17765), (17765, 17765, 22610), (22610, 22610, 27455), (27455, 27455, 32301)](test 段依次 ≈ 牛市尾/暴跌前/暴跌(iteration_v1 的 val 期)/横盘(其 test 期))

<!-- Analyst narrative appended below. -->

## 解读

- **主问题的答案:+0.465 没有幸存。** iteration_v1 的 val 段(2026-02→04 暴跌期)
  对应本 harness 的 fold 3:同一批事件在真正 OOS 评估下 ρ 只剩 **+0.219**,且下一折
  (横盘期)翻负(−0.151)。原 +0.465 的构成:该 split 同时被用于 early stopping
  (模型选择)+ 单一 regime 顺风。FACT:anchored 加权均值 ρ=+0.065(OOS n=140,
  噪声带 ±0.17)——不构成排序能力。
- **真正的新发现(OBSERVATION):anchored 在全部 4 折都逐折优于 legacy**
  (+0.085 vs −0.129 / +0.064 vs +0.029 / +0.219 vs +0.013 / −0.151 vs −0.249),
  配对符号检验 p≈0.06(4/4)。语义锚定(把决策点移到破位触发)对排序有一个
  微弱、跨 regime 一致的正向贡献——方向对,幅度不够。
- **池级(FACT)**:anchored all−ctrl −5.4bp、legacy +4.8bp,两臂事件池仍与同月
  同波动桶随机入场无异。
- HYPOTHESIS:6h 固定 horizon 的 short_utility 可能钝化了破位触发的优势
  (破位后的行情长短不一);若要继续,下一个单变量应当是标签/退出定义,
  而不是再调触发。

## 风险与诚实声明

- anchored fold 1 的 inner-val 只有 10 个事件,early stopping 在该折上近乎随机;
  已按 gate 最低线运行并如实报告。
- 两臂对比在同折同参数下严格受控;但 "4/4 优于" 基于 4 个相关样本(anchored 事件
  是 legacy episode 的子集变换),p≈0.06 只能算方向性证据。
- P02 对照组已用修复后的函数重算:模型指标逐位不变,池-对照差刷新为
  −3.9~+4.8bp(原 −6.7~+5.7bp),结论不变。
- holdout 0 消耗;本轮无任何参数扫描(两臂只跑了各一次)。

## 下一步选项(owner 决策)

- **A. 停止**:三轮 + 两次 walk-forward 复检后,诚实结论是——该信号族(MA 压缩
  → 无论何种时点/触发)在 ETH 15m、6h short_utility 下没有可确认的排序或池级
  edge;anchored 的语义改善真实但不足以变现。
- **B. 换标签再给一次机会**(单变量):保持 anchored 事件不动,把 6h 固定 horizon
  换成 first-hit barrier(TP/SL by ATR)——iteration_v1 任务书当时禁止,现在是
  自然的下一个变量;仍在 pre-holdout 数据上走同一 walk-forward。
- **C. 图廊裁决优先**:先翻 `reports/iteration_v1/review_gallery.html`,若 anchored
  的 decision 图在你眼里仍不是目标形态,A/B 都不用做。
