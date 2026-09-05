# v3.0：无模板内容神经网络

本分支：`release/v3.0-content`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v3_0_content.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

每条日志生成96位固定哈希内容序列；可学习64维Embedding经mean/max pooling得到128维，再经MLP输出内容表示。exp01仅raw序列，exp02增加结构塔，exp03用field序列替代raw；三分类融合头与内容辅助头共同训练。field_token_ids的上下文全集为format、action、protocol、event_code、event_name、http_method、dst_port_bucket、VPC log_status、JSON/XML/键值文本字段名，以及malware、potentially_harmful、threat、authentication、duo_push、no_response、invalid_passcode、powershell、process九类安全信号。它们按有限预算取入，不保证全部同时保留。

## 完整输入特征

exp01仅raw_token_ids；下面结构字段只用于exp02/exp03，分别附加raw_token_ids/field_token_ids。

### 类别输入（8个）

| 字段 | 定义 |
|---|---|
| `pipeline` | 采集管道；空串/NULL归为缺失类别。 |
| `product_name` | 产品名称；不直接决定威胁标签。 |
| `product_group` | 产品粗组：missing、asa、aws_vpc、other_suspicious_products（Precinct/Falcon）、other。 |
| `src_ip_kind` | 源地址外形类别：missing、ipv4_shape、host_token、other；仅按当前正则判外形。 |
| `port_bucket` | 源端口桶：missing、0–1023、1024–49151、49152以上；基于可解析整数。 |
| `message_length_bucket` | 正文长度桶：missing、1–120、121–180、181–300、301–1000、1001以上。 |
| `structure_combo` | pipeline、product_group、message_length_bucket和是否含deny拼接的类别。 |
| `network_missing_pattern` | 源IP/目的IP/源端口三个缺失位按顺序拼接，如111。 |

### 数值输入（23个）

| 字段 | 定义 |
|---|---|
| `src_port_number` | 原始src_port尝试转整数，缺失填-1。 |
| `src_ip_missing` | src_ip对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `dst_ip_missing` | dst_ip对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `src_port_missing` | src_port对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `src_host_missing` | src_host对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `dst_host_missing` | dst_host对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `username_missing` | username对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `product_missing` | product对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `message_missing` | message对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `network_present_count` | src_ip、dst_ip、src_port三个字段存在数量，0–3。 |
| `message_length` | 原始正文字符长度，空正文为0。 |
| `src_ip_length` | src_ip对应字符串字符长度，空值按空串。 |
| `dst_ip_length` | dst_ip对应字符串字符长度，空值按空串。 |
| `src_host_length` | src_host对应字符串字符长度，空值按空串。 |
| `dst_host_length` | dst_host对应字符串字符长度，空值按空串。 |
| `username_length` | username对应字符串字符长度，空值按空串。 |
| `message_has_deny` | 小写正文是否含deny子串；不是完整语义判断。 |
| `message_has_allow` | 小写正文是否含allow子串。 |
| `message_has_accepted` | 小写正文是否含accepted子串。 |
| `message_has_failed` | 小写正文是否含failed子串。 |
| `message_has_blocked` | 小写正文是否含blocked子串。 |
| `message_starts_angle` | 去左空白后的正文首字符是否为<。 |
| `message_contains_json` | 正文是否包含{；并不表示JSON解析成功。 |

## 训练、实验结果、缺陷与后续解决方法

固定哈希不是预训练语言模型；它把正文词、相邻词组合和字符片段转换成可学习Embedding索引。
原文归一化最多处理8,192字符，词筛选最多64个，使用首尾选取；生成96位置时raw最多40个词、
12个bigram，其余按字符片段循环填充。field表示优先加入最多24个上下文符号后再填词和字符，
不是完整无损JSON。hash=CRC32(symbol) mod (65536-2)+2；0为padding、1为空文本，
2–65535为哈希桶，允许碰撞。mean/max池化不保留完整词序，bigram只提供局部次序。

内容Embedding=64维；mean/max拼128维，经Linear128→LayerNorm→SiLU→Dropout0.15→
Linear128→SiLU得到128维。融合组再接结构塔与三分类头，内容辅助交叉熵权重0.25。
AdamW lr=0.001，weight_decay=1e-5，batch=2048，valid_batch=4096，
epochs最多12，patience=3，class_weight_power=0，token_dropout=0.05，
category_dropout=0.05，seed=20260828。完整输入字段见特征清单，field审计列不全是模型输入。

| 实验 | 最佳epoch | Score | Macro-F1 | 错误 | Log Loss | malicious召回 |
|---|---:|---:|---:|---:|---:|---:|
| exp01纯raw内容 | 3 | 0.903413323416246 | 0.875456762839011 | 7,567 | 0.024386343102812555 | 0.5037005408482779 |
| exp02结构+raw | 2 | 0.9762209762727161 | 0.9565217549231416 | 2,618 | 0.004142098408710442 | 0.8145459721036151 |
| exp03结构+field | 1 | 0.9113565658929034 | 0.9206922640920322 | 5,384 | 0.006224734148071435 | 0.6175633361799032 |

每轮Score完整留存如下，数组第一个数对应epoch1：

- exp01：0.8720943586、0.8754965965、0.9034133234、0.8839441089、0.9019810565、0.8737281575。
- exp02：0.9581935623、0.9762209763、0.9751146505、0.9726524286、0.8837811888。
- exp03：0.9113565659、0.8831131753、0.8831062163、0.8745470901。

exp02最佳轮的独立content辅助头Score仅0.8772237417、malicious召回0.3885567891；
exp03分别0.8553482423、0.3696270993。融合成绩不能当内容塔单独能力。
exp03有5,054条syslog deny恶意被判正常；字段插入会改变有限token预算和优化过程，
支持信息竞争的怀疑，但还需按字段移除/预算固定的对照才能确认唯一原因。

exp02的2618错分解为2330 malicious→suspicious、276 malicious→benign、
10 suspicious→benign、2 benign→suspicious。2330个子类型错都为vpc_flow/reject。
训练REJECT OK共10,620条，全部AWS VPC Security/suspicious；验证有1,853同源suspicious，
另有2,664个产品/厂商缺失的同语义malicious。模型把拒绝动作与业务子类型关联，
促使下一版把威胁检测和子类型分开；这也解释为什么仅提高malicious权重不够。

## 云平台运行

```bash
git fetch origin
git switch release/v3.0-content
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v3_0_content
nohup bash scripts/run.sh /root/work > artifacts/v3_0_content/nohup.log 2>&1 &
echo $! > artifacts/v3_0_content/train.pid
tail -f artifacts/v3_0_content/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
