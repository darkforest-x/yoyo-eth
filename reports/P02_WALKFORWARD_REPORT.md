# P02 — Walk-forward 触发点 × 池宽 网格实验

Owner 2026-08-12 批准的打包改动(1 触发点语义 2 扩池 3 walk-forward)。归因方式:
同一 harness 下的 2x3 网格,任一行/列内只有一个变量在动。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_walkforward.py --config configs/walkforward.yaml
python3 -m pytest tests/ -q
```

## Harness

| item | value |
|---|---|
| data | 2025-06-01 12:45:00+00:00 .. 2026-05-03 23:45:00+00:00, 32301 bars, gaps=0 (rows >= 2026-05-04 excluded right after CSV parsing, before all computation) |
| folds | 4 anchored, initial train 40%, OOS = last 60% in equal slices |
| embargo gap | 164 bars; horizon 24 bars |
| threshold freeze | per fold, inner-train bars only (inner val 15% of train, early stopping only) |
| top-decile | selected PER FOLD, outcomes pooled (fold models' scores are not cross-comparable) |
| control | matched random per fold test segment (month x ATR tercile, 20 draws/event), pooled |
| costs | 0.001 round-trip (SWAP_TAKER, owner value); sweep [0.001, 0.002] |

## 网格结果(全部 out-of-sample)

| cell | OOS n | rho per fold | rho wmean | top10 gross | top10 net@0.001 / top10 net@0.002 | all gross | control gross | all-ctrl | top10-ctrl |
|---|---|---|---|---|---|---|---|---|---|
| zone_start@0.3 | 195 | [-0.132, 0.202, -0.04, -0.225] | -0.035 | -17.7bp | -28bp / -38bp | +14.1bp | +9.3bp | +4.8bp | -27.0bp |
| zone_start@0.45 | 177 | [0.052, -0.083, 0.011, 0.255] | +0.052 | +31.0bp | +21bp / +11bp | +7.8bp | +11.7bp | -3.9bp | +19.3bp |
| dispersion_exit@0.3 | 207 | [0.025, 0.006, 0.015, 0.117] | +0.037 | +60.7bp | +51bp / +41bp | +2.8bp | +4.7bp | -1.9bp | +56.0bp |
| dispersion_exit@0.45 | 192 | [-0.085, -0.135, 0.184, -0.011] | -0.012 | +48.9bp | +39bp / +29bp | +9.3bp | +8.2bp | +1.1bp | +40.6bp |
| price_breakout@0.3 | 270 | [-0.072, 0.029, -0.273, -0.103] | -0.096 | -34.8bp | -45bp / -55bp | +8.2bp | +8.3bp | -0.1bp | -43.1bp |
| price_breakout@0.45 | 335 | [0.003, -0.127, -0.084, -0.023] | -0.059 | +1.9bp | -8bp / -18bp | +5.8bp | +8.9bp | -3.1bp | -7.0bp |

triggers: zone_start = 压缩带开头开火(MVP 原版,对照); dispersion_exit = 带向上穿出
(压缩结束); price_breakout = 收盘首次离开 [ma_lower, ma_upper](突破尝试)。
quantile 0.30 = MVP 原版(对照); 0.45 = 扩池。

<!-- Narrative sections are appended by the analyst. -->

## 解读

- **统计力已解决**:每格 OOS 177–335 个事件(MVP 是 40),OOS 覆盖数据后 60%、
  横跨牛市/暴跌/横盘三种 regime——MVP 的两个设计缺陷(样本量、单切分撞 regime)
  都已消除。
- **主结论:六个格没有一格的排序能力超出噪声**。加权 spearman 全部落在
  −0.096 ~ +0.052(pooled n≈200 的噪声带 ±0.14);MVP val 上的 +0.17 在 walk-forward
  下不复现,确认为噪声。
- **池级 beta 修正**:all-vs-control 收敛到 −6.7 ~ +5.7bp——MVP 报的"池子跑输随机
  35bp"是 val/test 时段特异,拉长到 60% 数据后,压缩池与随机入场无显著差异
  (即:既没有负 edge,也没有正 edge)。
- **触发点归因(1 号变量)**:同 quantile 下对比 top10 毛收益,dispersion_exit
  (+60.7 / +48.9bp)一致优于 zone_start(−17.7 / +31.0bp)和 price_breakout
  (−34.8 / +1.9bp)。方向上支持"决策点应放在压缩结束而非开始"的假设;
  price_breakout 表现最差,说明宽松定义下的收盘破带多为假突破。
- **dispersion_exit top10 苗头未过显著性**:组内置换检验(fold 内同规模随机选取,
  5000 次)p=0.044(@0.30)/ 0.113(@0.45),都未达项目 p<0.01 标准;两格事件池
  高度重叠,不是独立证据;fold 间 top10 毛收益 −53 ~ +177bp 不稳定。且"挑最好
  的格子做检验"本身带事后选择色彩,真实 p 值应更差。**记录为未确认苗头,
  不构成 edge 声明。**
- **扩池归因(2 号变量)**:q0.30→0.45 没有系统性改善排序或 top10;扩池只解决了
  样本量,没带来信息。

## 风险与诚实声明

1. zone_start 两格的 `compression_duration` 特征恒等于 3(去重取 zone 首个达标
   bar 的必然结果,MVP 同款退化),该两格实际有效特征数 26;exit 类触发该特征
   有效(3–95)。"六格特征集相同"的归因表述有此细微差异。
2. 复核图的 top/bottom 排序已改为 fold 内百分位(fold 模型分数不可跨 fold 比较);
   图标题中的 pred 值为百分位而非原始分。
3. 折内 early-stopping 的 val 标签窗原可伸进 test 段前 24 根,审查后已右缩
   horizon 修复;修复前后网格数字无实质变化。
4. funding 成本仍未建模;dispersion_exit 的 NaN 防御语义为"宁漏检不造事件"
   (真实数据 warmup 段不可达)。
5. 置换检验只对 dispersion_exit 两格做了(事后选择),其余格未检——因为它们
   连方向性苗头都没有。
6. holdout(≥2026-05-04)本轮继续 0 消耗;所有结果均为 pre-holdout OOS。
7. Owner 授权记录:触发点+扩池+walk-forward 打包改动,owner 2026-08-12 对话中
   批准("1 2 3");归因由网格结构保留。

## 下一步选项(需 owner 决策)

- **A. 到此为止**:两轮实验的诚实结论是——宽松 MA 压缩池(无论在带内哪个时点
  开火)在 ETH 15m 上没有可检出的做空排序 edge;dispersion_exit 的 top10 苗头
  未过显著性。
- **B. 一次性确认实验**:冻结 dispersion_exit@0.30 当前全部参数,在 holdout
  (≥2026-05-04,约 8,980 根)上做单次预注册式验证。这是正式 holdout 消耗
  (该数据段第 1 次),需要你明确批准;通过标准建议事先定死(top10 净收益>0
  且置换 p<0.05)。
- **C. 精化 exit 语义再迭代**(如 zone_length 下限、突破方向过滤、结合
  Local Signal V2 的平台判据):继续挖会引入多重比较,建议每轮预注册一个变体。

(P02 结束即停,未进入任何后续阶段。)
