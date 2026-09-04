# V10：V7锚定的内容证据救援实验

## 1. 为什么从metadata改为content

V9在seed29锚点上的1011条“最终benign、metadata threat”冲突全部是真实benign，
剩余32条威胁漏报没有一条属于该候选集。因此V9正确回退到epoch 0，同时否定了继续放大
metadata冲突的假设。

对32条漏报拆解后发现另一种反向冲突：

```text
20条 malicious JSON：
  anchor threat平均=0.438180
  metadata threat平均=0.003641
  content threat平均=0.947359

1条 Cisco Duo suspicious：
  anchor threat=0.253372
  metadata threat=0.000602
  content threat=0.938998
```

这些日志的内容分支已经识别出威胁，但最终融合层被低metadata证据压回benign。V10只研究
这一类“最终benign、content threat”冲突。

## 2. 为什么不直接使用0.94阈值

验证集阈值审计结果为：

```text
content阈值  候选  真威胁  真benign  精度
0.50         294     21       273    0.071429
0.90          41     21        20    0.512195
0.94          20     16         4    0.800000
0.96           9      8         1    0.888889
```

直接选择0.94或0.96会使用验证标签调规则，存在泄漏，而且仍会制造FP。V10不把这些阈值
写进推理逻辑。候选条件只要求冻结content分支的logit margin大于0；是否相信content由
训练数据学习。

## 3. 模型约束

```text
冻结seed29 V7-H2
├── anchor最终threat logits
├── metadata辅助threat logits
├── content辅助threat logits
└── 冻结语义与内容表示
          │
          └── 仅当anchor=benign、content=threat时形成候选
                       │
                       ▼
               零初始化可信度残差
```

修正量为：

```text
delta = content_conflict_mask
      × max(0, tanh(learned_trust_logit))
      × clamp(content_margin - anchor_margin, 0, 24)
```

硬约束：

1. epoch 0逐行等于seed29锚点；
2. 只允许将anchor-benign向threat方向移动；
3. 最大只移动到content分支的logit证据，不无界放大；
4. 不修改anchor已判threat的行，所以24条subtype错误和已有1条FP不属于本实验；
5. 每轮训练只有competition score更高，或同分时错误/Log Loss更优，才替换epoch 0。

## 4. 训练样本

训练不读取验证标签：

- 正例：训练集中真实为threat且冻结content分支投threat的行；
- 困难负例：训练集中content margin最高的benign行；
- 默认困难负例上限为正例的2倍；
- 可信度输入包括冻结metadata、semantic、raw-content表示、两个分支margin、新颖度和
  组合频次；
- CR2额外加入head/middle/tail/key-value多视图表示。

验证集标签只用于正常的epoch选择和最终评估，不用于构造阈值、日志关键字或厂商规则。

## 5. 两组实验

| 运行 | 可信度输入 | 目的 |
|---|---|---|
| `cr1_anchor_content` | seed29已有冻结表示 | 判断已有表示能否识别真假content冲突 |
| `cr2_multiview_content` | CR1 + 四视图内容 | 判断长日志局部内容能否区分failure与success |

## 6. 云端运行

```bash
git pull --ff-only origin feat/v9-anchored-conflict-residual

mkdir -p artifacts/v10_content_rescue
nohup env \
  V10_PROCESSED_ROOT=/root/soc-threat-detection-sf-2026-02-1/data/processed \
  V10_ANCHOR_MODEL=artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/model.pt \
  V10_ANCHOR_PREDICTIONS=artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/valid_predictions.parquet \
  V10_ANCHOR_METRICS=artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/metrics.json \
  bash scripts/run_cloud_v10_content_rescue.sh /root/work \
  > artifacts/v10_content_rescue/nohup.log 2>&1 &
echo $! > artifacts/v10_content_rescue/train.pid
tail -f artifacts/v10_content_rescue/nohup.log
```

已有V6/V8/V9特征会校验并复用；中断后重复同一条命令可以恢复。

## 7. 结果检查

```bash
cat artifacts/v10_content_rescue/comparison.json
cat artifacts/v10_content_rescue/cr1_anchor_content/analysis/residual_summary.json
cat artifacts/v10_content_rescue/cr2_multiview_content/analysis/residual_summary.json
```

重点比较：

```text
anchor_threat_errors = 33
final_threat_errors < 33
fixed_anchor_threat_errors > 0
new_threat_errors 尽量为0
competition_score > 0.9993387401713992
```

总三分类错误不一定同步减少：把 `malicious -> benign` 改为
`malicious -> suspicious` 仍算一条三分类错误，但已经修复威胁二分类漏报，并会提升竞赛分数。

