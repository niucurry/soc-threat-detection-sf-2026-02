# V3：无模板内容神经网络

v3.0 的目标是检验一个神经网络能否学习日志正文，而不是根据 `template_id`、`schema_id`、
`semantic_template_id`、Drain cluster 或完整消息 ID 查表。这些字段全部不进入模型。

正文先屏蔽时间戳、UUID、邮箱、IPv4/IPv6、脱敏 USER/ORG/CRED/HOST 实体、长十六进制和
长数字，保留 deny、reject、failure、malware、process 等安全词。`raw_token_ids` 包含：

1. 归一化单词；
2. 相邻单词 bigram；
3. 每个词的首部、中部、尾部字符 3/4/5-gram 抽样。

每个符号用固定 CRC32 映射到 65,536 桶，每行最多 96 个 token。固定哈希不在验证集拟合
词表。`field_token_ids` 在同一内容上再加入全部这些上下文类型：`message_format`、
`semantic_action`、`network_protocol`、`event_code`、`event_name`、`http_method`、
`dst_port_bucket`、安全信号（malware、potentially_harmful、threat、authentication、
duo_push、no_response、invalid_passcode、powershell、process）、VPC log status，以及从 JSON/
XML/键值文本抽取的字段名。

内容编码器为 token Embedding，分别做 mean/max pooling，再经两层 MLP 得到 128 维向量。
三个实验是同一版本的消融：

| 实验 | 输入 | Score | 错误 | malicious recall |
|---|---|---:|---:|---:|
| v3.0-exp01 | `raw_token_ids`，仅内容 | 0.9034133234 | 7,567 | 0.5037005408 |
| v3.0-exp02 | v1.0 全部结构输入 + `raw_token_ids` | 0.9762209763 | 2,618 | 0.8145459721 |
| v3.0-exp03 | v1.0 全部结构输入 + `field_token_ids` | 0.9113565659 | 5,384 | 0.6175633362 |

exp02 最好，证明结构和内容互补；exp03 下降说明把许多显式字段塞进 96-token 序列会挤占
正文。exp02 的错误里有 2,330 条 `malicious -> suspicious`，集中在 VPC `REJECT OK`：
模型已经知道它是威胁，却把内容语义和业务子类型绑定。下一版将“是否威胁”和“是哪种威胁”
拆成两个任务。

入口：`scripts/run_cloud_v3_0_content.sh`。
