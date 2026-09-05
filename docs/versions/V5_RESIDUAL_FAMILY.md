# V5：冻结锚点的冲突残差

V5 冻结可复现的 `v4.0-exp02-seed20260829` 全部权重。残差只处理一种候选：锚点最终判
benign，但指定独立分支判 threat。最后一层严格零初始化，所以 epoch 0 的逐行概率等于
锚点；候选外样本不能改变；训练 epoch 未严格优于锚点时自动回退 epoch 0；subtype logits
始终来自冻结锚点。

可信度输入完整包含：冻结 metadata 向量 128 维、semantic 向量 64 维、raw content 向量
128 维、metadata threat margin、content threat margin、novelty gate、
`log1p(semantic_combo_count)`。最终 anchor margin 故意不输入，避免可信度头简单复制锚点。
多视图实验额外输入 128 维四视图向量；其共享 token 编码器从锚点复制初始化，但复制后的
编码器和 view fusion 都会训练，只有原锚点冻结。

残差公式：

```text
candidate = (anchor_margin < 0) AND (evidence_margin > 0)
gap       = clamp(evidence_margin - anchor_margin, 0, max_conflict_gap)
delta     = candidate × max(0, tanh(trust_logit)) × stop_gradient(gap)
new_margin = anchor_margin + delta
```

## v5.0：metadata evidence

候选 1,011 行全部是真 benign，32 个锚点 FN 没有一条进入候选；两个实验都自动选择 epoch 0，
Score 和 57 错完全等于锚点。这是否定结果：metadata 分支无法救剩余漏报。

## v5.1：content evidence，gap=24

32 个 FN 中有 21 个属于“content 强烈报警、融合压回 benign”。content 候选共有 294 行，
其中 21 个真威胁、273 个 benign。训练能修复全部 21 个 FN，但 gap=24 允许内容分支推翻
锚点极高置信 benign：exp01 变成 43 FP/11 FN/78 错/Score `0.9994966836`；多视图 exp02
为 63 FP/11 FN/98 错/`0.9994142658`。方向有效，修正幅度过大。

## v5.2：content evidence，局部可信域 gap=2

gap=2 意味着残差最多把 threat odds 乘以 `exp(2)≈7.39`；anchor threat 概率低于约
0.119 的样本即使 content 很高也不能直接翻转。它不包含厂商硬规则。

正式默认 `v5.2-exp01-anchor-content-gap2`：Score `0.9996698618`、36 个三分类错误、FP=1、
FN=11、子类型混淆=24、相对锚点修正 21 且新增错误 0、Log Loss `0.0002189643`。修复的是
20 条产品/厂商缺失的 malicious JSON 和 1 条 Cisco Duo suspicious JSON。exp02 多视图的
hard prediction 完全相同，Log Loss `0.0002179833`，但计算和特征成本更高，因此不作为默认。

剩余 36 错是三个独立问题：1 个边界 benign→threat、11 个所有分支都偏 benign 的 FN、
24 个 malicious→suspicious。下一阶段应先用训练集时间外推/OOF 校准验证 gap=2，而不是在
同一验证集继续扫描 gap；之后为 24 个错误设计只作用于已判 threat 的 subtype 残差。

入口：`scripts/run_cloud_v5_anchored_residual.sh`。默认只跑正式 exp01；设置
`V5_RUN_MULTIVIEW=1` 才运行 exp02。
