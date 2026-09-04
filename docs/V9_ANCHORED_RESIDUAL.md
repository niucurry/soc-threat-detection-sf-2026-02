# V9：V7锚定的分支冲突残差实验

## 1. 实验依据

V8正式结果表明：

- A1用四视图整体替换原内容塔后，错误由46增加到208；
- B1保留原内容塔并加入全局正例证据损失后，原46条FN一条未修复，新增10条FP；
- C1四视图加证据损失后，原46条FN仍未修复，新增20条FP；
- C1的20条FP全部是 `benign -> suspicious`，问题位于第一层威胁门；
- C1错误日志的四个视图大多已经满64个token，不能再把失败解释为简单截断。

概率审计进一步显示两种不同情况：

1. 6条Windows deny日志中，元数据分支威胁概率约0.972，最终威胁概率约0.058，属于明确
   的分支冲突；
2. 其余主要FN中，最终、元数据和内容分支都接近benign，属于训练证据不足，不能靠融合
   规则凭空修复。

因此V9只研究第一类可修复冲突，不尝试用规则覆盖第二类样本。

## 2. 模型结构

```text
冻结的V7-H2
├── 原96-token内容塔：完全保留
├── 第一层threat logits：作为锚点
├── 第二层subtype logits：完全保留
└── 元数据辅助threat logits
             │
             └── 仅当“最终=benign、元数据=threat”时形成冲突候选
                         │
                         ▼
                可信度残差（初始严格为0）
                         │
                         ▼
                最终threat logits
```

残差修正量为：

```text
delta = conflict_mask
      × max(0, tanh(learned_trust_logit))
      × clamp(metadata_margin - anchor_margin, 0, 24)
```

它有四项硬约束：

1. 最后一层零初始化，epoch 0的概率和分类逐行等于V7；
2. 只有V7最终判benign、元数据分支判threat时才能增加威胁概率；
3. 最大修正只到元数据分支的logit，不允许无界放大；
4. malicious/suspicious子类型logits直接使用冻结V7，不参与训练。

## 3. 可信度如何训练

V7在自己的训练集上通常没有最终/元数据冲突，不能直接从训练错误学习。V9改为：

- 正例：训练集中标签为威胁、且元数据分支判为threat的样本；
- 困难负例：训练集中元数据威胁margin最高、最接近边界的benign样本；
- 默认困难负例数不超过正例数的2倍；
- 可信度头用训练标签学习，不读取验证标签；
- 推理时即使可信度头给出正分，也仍必须先满足严格冲突掩码。

这一步是为了学习“元数据证据何时可靠”，不是把 `deny`、`fail` 或厂商名称写成标签规则。

## 4. 两组对照

| 运行 | 可信度输入 | 目的 |
|---|---|---|
| R1 `r1_anchor_conflict` | 冻结V7的元数据、语义、原内容表示和分支margin | 判断已有表示能否识别可靠冲突 |
| R2 `r2_multiview_conflict` | R1全部输入 + head/middle/tail/key-value | 判断长内容是否帮助区分真假冲突 |

R2的多视图共享token编码器从V7原内容编码器复制参数，只新训练视图融合和可信度头，避免
重演V8-A1的从头替换问题。

## 5. 安全回退

训练前先评价epoch 0并把它列为正式候选。后续epoch只有在下列顺序上更优才会替换：

1. competition score更高；
2. score相同时错误更少；
3. score和错误都相同时log loss更低。

如果所有训练epoch都退化，最终 `model.pt` 和 `valid_predictions.parquet` 自动回退到
epoch 0，也就是精确V7结果。

## 6. 云平台命令

```bash
git fetch origin
git switch --track origin/feat/v9-anchored-conflict-residual
git pull --ff-only

mkdir -p artifacts/v9_anchored_residual
nohup bash scripts/run_cloud_v9_anchored_residual.sh /root/work \
  > artifacts/v9_anchored_residual/nohup.log 2>&1 &
echo $! > artifacts/v9_anchored_residual/train.pid
tail -f artifacts/v9_anchored_residual/nohup.log
```

脚本要求已有：

```text
artifacts/v7_hierarchical_content/h2_hierarchical_novelty/model.pt
```

如果旧云实例只保留了V7指标或预测而缺少 `model.pt`，先更新本分支，再重新运行V7脚本。
V7脚本现在只有在模型、指标和预测三个文件都完整时才跳过；H1完整时会复用，只重训缺失的
H2：

```bash
mkdir -p artifacts/v7_hierarchical_content
nohup bash scripts/run_cloud_v7_hierarchical_content.sh /root/work \
  > artifacts/v7_hierarchical_content/recover.log 2>&1 &
tail -f artifacts/v7_hierarchical_content/recover.log
```

V6、V8特征和V8分片存在时会直接复用。V9只额外执行一次按 `event_id` 的V6/V8连接，不会
重新解析全部长日志。云平台中断后重复同一条命令即可；已有 `metrics.json` 的运行会跳过。

## 7. 结果回传

```bash
cat artifacts/v9_anchored_residual/comparison.json

for run in r1_anchor_conflict r2_multiview_conflict; do
  echo "===== ${run} metrics ====="
  cat "artifacts/v9_anchored_residual/${run}/metrics.json"
  echo "===== ${run} residual ====="
  cat "artifacts/v9_anchored_residual/${run}/analysis/residual_summary.json"
  echo "===== ${run} errors ====="
  cat "artifacts/v9_anchored_residual/${run}/analysis/error_summary.json"
done
```

重点检查：

```text
epoch_zero_is_exact_v7_anchor = true
anchor_errors = 46
fixed_anchor_errors > 0
new_errors 尽量为0
final_errors < 46
competition_score > 0.9993140394
```

如果 `best_epoch=0`，表示可信度模型没有找到能在不增加更多错误的前提下修复V7的方法；
这仍是有效结果，而且输出不会比V7差。
