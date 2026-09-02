# V6 内容学习神经网络实验

## 实验目标

V6 验证模型能否学习日志中的动作、对象、结果、原因和事件内容，
而不是记住 Drain 模板或 schema 编号。V6 神经网络的输入中明确没有：

```text
template_id
schema_id
semantic_template_id
drain_cluster_id
message_template
```

## 内容编码

首先将不稳定实体归一化：

```text
IP             -> ip_token
UUID           -> uuid_token
时间戳         -> timestamp_token
USER-*         -> user_token
HOST-*         -> host_token
ORG-*          -> org_token
CRED-*         -> cred_token
长数字/长十六进制 -> number_token / hex_token
```

安全语义词保留，例如：

```text
deny blocked rejected malware potentially harmful
authentication duo_push invalid_passcode no_response
powershell process logon_failure
```

每行最多保留 96 个哈希 token，包括：

- 单词；
- 相邻双词；
- 3～5 字符 n-gram；
- 字段名；
- 稳定事件语义，如 `action_reject`、`event_code_4625`。

哈希函数固定，没有在训练或验证数据上拟合词表，因此没有未登录
词映射到随机 Embedding 的问题。0 号是固定零向量，只用于 padding。

VPC Flow 中的：

```text
REJECT OK
```

被明确表示为：

```text
action_reject
log_status_ok
```

`OK` 不会再被错误理解为安全事件成功。

## 模型结构

```text
V1 稳定结构特征 -> Structured Tower --\
                                           -> Fusion Head -> 三分类
日志内容哈希 token  -> Content Tower -----/
```

Content Tower 使用带 `padding_idx=0` 的 Embedding，然后进行 mean/max
pooling。融合模型同时优化：

```text
Loss = 融合头分类损失 + 0.25 * 内容头分类损失
```

辅助损失用于防止模型完全忽略日志内容。

## 实验组

| 输出目录 | 内容 |
|---|---|
| `e1_content_raw` | 只使用归一化日志内容 |
| `e2_fusion_raw` | V1 结构特征 + 归一化原文 |
| `e3_fusion_field` | V1 结构特征 + 字段名 + 内容 + 粗粒度语义 |

V4 是模板基线，V5.1 是 schema/语义模板重度组合的反例。若其指标文件
仍在默认位置，V6 会自动加入汇总。

## 云平台运行

```bash
git fetch origin
git switch feat/v6-content-neural
git pull --ff-only

mkdir -p artifacts/v6_content_neural

nohup bash scripts/run_cloud_v6_content_neural.sh /root/work \
  > artifacts/v6_content_neural/nohup.log 2>&1 &

echo $! > artifacts/v6_content_neural/train.pid
tail -f artifacts/v6_content_neural/nohup.log
```

预处理分片位于：

```text
data/processed/v6/train_content_shards
data/processed/v6/valid_content_shards
```

每个分片先写临时文件，完成后再原子替换。如果云环境关机，重新执行上述
`nohup` 命令：

- 完整分片会被校验并跳过；
- 损坏或不完整分片会自动重写；
- 已经有 `metrics.json` 的实验会跳过；
- 只会重跑未完成的模型。

中断恢复时不要设置 `V6_FORCE_PREPARE=1`。只有主动修改了 token 规则或
参数，需要删除全部 V6 分片重建时才使用该选项。

如果 NPU 显存不足：

```bash
V6_BATCH_SIZE=1024 V6_VALID_BATCH_SIZE=2048 \
  bash scripts/run_cloud_v6_content_neural.sh /root/work
```

## 结果回传

```bash
cat artifacts/v6_content_neural/comparison.json
cat data/processed/v6/v6_manifest.json

cat artifacts/v6_content_neural/e1_content_raw/metrics.json
cat artifacts/v6_content_neural/e2_fusion_raw/metrics.json
cat artifacts/v6_content_neural/e3_fusion_field/metrics.json

cat artifacts/v6_content_neural/e3_fusion_field/analysis/error_summary.json
wc -l artifacts/v6_content_neural/e3_fusion_field/analysis/error_rows.csv
head -30 artifacts/v6_content_neural/e3_fusion_field/analysis/error_rows.csv
```

主要验收标准是：

```text
competition_score > 0.9990226566
错误数 < 76
malicious recall > 0.9953031597
benign -> threat 仍为 0
fixed_by_candidate > new_in_candidate
```
