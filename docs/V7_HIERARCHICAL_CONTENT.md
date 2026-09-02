# V7 分层内容神经网络实验

## 为什么要改成分层分类

V6 E2 的 2,618 个错误中，2,330 个是：

```text
真实 malicious -> 预测 suspicious
content_family = vpc_flow
content_action = reject
```

这些行仍被正确识别为威胁，问题只出在威胁子类型。训练数据中的
`REJECT OK` 全部来自 AWS 产品且标注为 suspicious；验证数据新增了
“产品缺失 + REJECT OK + malicious”的组合。单个三分类头会把正文中的
`REJECT` 与 suspicious 直接绑定。

V7 将任务改为：

```text
                          -> benign
metadata + semantics + content -> Threat Head
                          -> threat -> Subtype Head -> malicious/suspicious
```

最终概率为：

```text
P(benign)     = P(benign)
P(malicious)  = P(threat) * P(malicious | threat)
P(suspicious) = P(threat) * P(suspicious | threat)
```

## 三类输入各自的职责

### Metadata Tower

使用 V1 稳定结构特征，并增加 vendor：

```text
pipeline, product_name, vendor_name, product_group
字段缺失状态、网络字段形态、消息长度和少量稳定标志
```

它产生威胁辅助预测，同时是子类型判断的基础模型。

### Semantic Tower

解析得到的字段独立编码，不再挤占正文 token：

```text
content_family
content_action
content_protocol
content_event_code
content_has_threat
content_has_authentication
content_has_potentially_harmful
```

这些字段只能作为模型输入，不会直接覆盖预测结果。

### Content Tower

继续使用 V6 已验证的 `raw_token_ids`：

- 屏蔽 IP、UUID、时间戳和脱敏实体；
- 保留单词、相邻双词和字符 n-gram；
- 固定 65,536 个哈希桶；
- 不在验证集上拟合词表；
- 不包含模板、schema 或聚类编号。

## 子类型残差与未见组合门控

子类型输出为：

```text
Subtype Logits
  = Metadata Subtype Logits
  + novelty_gate * Semantic/Content Residual
```

训练集只统计下面组合的出现次数：

```text
product_name + content_family + content_action
```

门控值为：

```text
count / (count + 32)
```

因此：

- 高频已见组合接近 1，允许内容修正子类型；
- 稀有组合影响较小；
- 未见组合等于 0，只使用元数据子类型模型。

频次统计完全不读取标签，也不读取验证集。它不是 target encoding，
也不是 template ID。内容残差的最后一层从零初始化，训练刚开始时不会
随机破坏元数据判断。

## 损失函数

```text
Loss = Threat Loss
     + 0.75 * Subtype Loss（仅真实威胁行）
     + 0.15 * Metadata Threat Auxiliary Loss
     + 0.25 * Content Threat Auxiliary Loss
     + 0.35 * Metadata Subtype Auxiliary Loss
```

Threat Loss 默认使用轻量类别权重 `power=0.25`。子类型默认不加类别
权重，避免再次把 malicious/suspicious 比例变化放大。

## 两组对照实验

| 输出目录 | 说明 |
|---|---|
| `h1_hierarchical_raw` | 分层分类，但所有子类型内容残差均开启 |
| `h2_hierarchical_novelty` | 分层分类，并按训练输入组合频次门控残差 |

H1 用来判断收益是否仅来自分层损失，H2 用来单独验证未见组合门控。

## 云平台运行

```bash
git fetch origin
git switch --track origin/feat/v7-hierarchical-content
git pull --ff-only

mkdir -p artifacts/v7_hierarchical_content

nohup bash scripts/run_cloud_v7_hierarchical_content.sh /root/work \
  > artifacts/v7_hierarchical_content/nohup.log 2>&1 &

echo $! > artifacts/v7_hierarchical_content/train.pid
tail -f artifacts/v7_hierarchical_content/nohup.log
```

V7 直接复用并校验：

```text
data/processed/v6/v6_train.parquet
data/processed/v6/v6_valid.parquet
```

已有完整 V6 特征时不会重新计算正文。云环境中断后重新运行同一条
`nohup` 命令即可，已有 `metrics.json` 的模型会跳过。

不要在恢复任务时设置 `V7_FORCE_PREPARE=1` 或 `V7_FORCE_TRAIN=1`。

如果 NPU 显存不足：

```bash
nohup env V7_BATCH_SIZE=1024 V7_VALID_BATCH_SIZE=2048 \
  bash scripts/run_cloud_v7_hierarchical_content.sh /root/work \
  > artifacts/v7_hierarchical_content/nohup.log 2>&1 &
```

## 结果回传

```bash
cat artifacts/v7_hierarchical_content/comparison.json

cat artifacts/v7_hierarchical_content/h1_hierarchical_raw/metrics.json
cat artifacts/v7_hierarchical_content/h2_hierarchical_novelty/metrics.json

cat artifacts/v7_hierarchical_content/h2_hierarchical_novelty/analysis/error_summary.json
wc -l artifacts/v7_hierarchical_content/h2_hierarchical_novelty/analysis/error_rows.csv
head -30 artifacts/v7_hierarchical_content/h2_hierarchical_novelty/analysis/error_rows.csv
```

重点检查：

1. `subtype_confusion` 是否显著少于 V6 E2 的 2,330；
2. `threat_false_negative` 是否不高于 286；
3. `threat_false_positive` 是否仍接近 2；
4. H2 的 `unseen_combo_errors` 是否低于 H1；
5. H2 是否修复“产品缺失 + vpc_flow + reject”。
