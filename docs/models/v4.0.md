# v4.0：威胁与子类型分层神经网络

本分支：`release/v4.0-hierarchical`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v4_0_hierarchical.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

元数据、语义、raw内容三塔分别128/64/128维。先预测benign/threat，再用metadata子类型头与semantic/content残差判断malicious/suspicious。概率为P(m)=P(t)P(m|t)、P(s)=P(t)P(s|t)，决策先按threat阈值再取子类型最大值。频次门控count/(count+32)只限制子类型残差。

## 完整输入特征

### 类别输入（9个）

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
| `vendor_name` | 厂商名称；空值单独编码。 |

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

### 语义塔完整输入

### 类别输入（4个）

| 字段 | 定义 |
|---|---|
| `content_family` | 内容解析家族。 |
| `content_action` | 内容粗动作。 |
| `content_protocol` | 内容协议。 |
| `content_event_code` | 内容事件码。 |

### 数值输入（4个）

| 字段 | 定义 |
|---|---|
| `content_has_threat` | 内容安全词组的威胁信号；不是标签。 |
| `content_has_authentication` | 认证关键词信号。 |
| `content_has_potentially_harmful` | potentially harmful相关词组信号。 |
| `raw_token_count` | raw序列非padding数量，上限96。 |

另输入raw_token_ids（96位）；v4.1多视图实验用multiview_token_ids（head/middle/tail/key_value各64位）替代raw。频次键为product_name、content_family、content_action。

## 训练、实验结果、缺陷与后续解决方法

metadata塔是类别Embedding+数值→256→LayerNorm→SiLU→Dropout0.12→128→LayerNorm→SiLU；
semantic塔对应96→64。content塔沿用raw内容128维。
威胁头拼320维→160→LayerNorm→SiLU→Dropout0.12→2。
metadata/content各有独立2维威胁辅助头。
metadata子类型头128→64→SiLU→2，semantic+content残差192→64→SiLU→2；
残差输出层零初始化。count门控只作用子类型残差，不控制第一层威胁头。

训练AdamW lr=0.0007、weight_decay=1e-5、batch=2048、valid_batch=4096、
epochs≤10、patience=4、threat权重power=0.25、subtype权重power=0、
token_dropout=0.05、category_dropout=0.02、novelty_pseudocount=32，
subtype loss=0.75、metadata-threat辅助=0.15、content-threat辅助=0.25、
metadata-subtype辅助=0.35。subtype及其辅助损失只在真实threat行计算。

| 实验 | Score | 错误 | FP | FN | subtype | Log Loss | epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp01不门控 | 0.9993140394289346 | 46 | 0 | 46 | 0 | 0.00032644834755403787 | 1 |
| exp02 count门控 | 0.9993140394289346 | 46 | 0 | 46 | 0 | 0.0002620398425025378 | 1 |

两组Macro-F1=0.999527322822607，Balanced Accuracy=0.9990635758890131。
矩阵[[1959573,0,0],[36,14016,0],[10,0,40417]]。
分类相同只支持门控改变概率损失，不能据此断言已完成概率校准（还需可靠性图/ECE）。
主要收益与分层改造同时出现，但单头到分层还伴随其他训练变化，严格因果归因仍需配对消融。

46个FN的全分组：

| 组 | 行数 | 诊断 |
|---|---:|---|
| 厂商产品缺失、长JSON认证 | 28 | content概率0.127–0.380，最终0.002–0.004 |
| 厂商产品缺失、Windows JSON、deny、事件500、ICMP | 6 | metadata约0.957，最终0.011–0.022 |
| 厂商产品缺失、Windows4672 | 2 | 两个辅助头均低 |
| Symantec DLP、4688、success | 8 | 未见组合，辅助头低 |
| Symantec DLP、1309 | 1 | 辅助头低 |
| Cisco Duo认证JSON | 1 | content约0.133，最终约0.00198 |

46行raw_token_count均96，45行长度>1000；这提示内容覆盖问题，但不证明决定字段一定被截断。
验证未见组合602,626行、错误9，错误率约0.00149%；已见约0.00262%。
未见组合没有更高错误率，故不继续单独调count门控。

| threat阈值 | Score | 错误 | FP | FN |
|---:|---:|---:|---:|---:|
| 0.50 | 0.9993140394 | 46 | 0 | 46 |
| 0.40 | 0.9993140394 | 46 | 0 | 46 |
| 0.30 | 0.9993140394 | 46 | 0 | 46 |
| 0.20 | 0.9993140394 | 46 | 0 | 46 |
| 0.10 | 0.9992645421 | 58 | 12 | 46 |
| 0.05 | 0.9992191794 | 69 | 23 | 46 |
| 0.02 | 0.9989184424 | 141 | 97 | 44 |
| 0.01 | 0.9914071626 | 1,731 | 1,691 | 40 |

降至0.2未修任何FN，0.02只修2个却增加97个FP，故不采用全局低阈值。

46错checkpoint丢失后，首次恢复57错/Score0.9991498166/epoch2，FP1、FN56、
subtype0；其中新增10个syslog deny漏报及1个Falcon误报。两份特征文件SHA不同，
只能确定字节不同，不能仅凭哈希判定实际特征值不同；行顺序与Parquet编码也会改变文件哈希。
固定留存的同一份内容特征后做四个seed：

| seed | Score | 错误 | FP | FN | subtype | Log Loss | epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260828 | 0.9992500022094916 | 50 | 0 | 50 | 0 | 0.0005319748723099184 | 1 |
| 20260829 | 0.9993387401713992 | 57 | 1 | 32 | 24 | 0.0002222812621389772 | 2 |
| 20260830 | 0.9771118717927942 | 1,428 | 6 | 1,422 | 0 | 0.0020848522449002094 | 5 |
| 20260831 | 0.9988966897809729 | 76 | 0 | 68 | 8 | 0.0003768637450844391 | 1 |

四个矩阵分别为：

1. [[1959573,0,0],[40,14012,0],[10,0,40417]]
2. [[1959572,0,1],[22,14006,24],[10,0,40417]]
3. [[1959567,0,6],[1412,12640,0],[10,0,40417]]
4. [[1959573,0,0],[58,13986,8],[10,0,40417]]

seed28和29共有32错；29修复28的18错，新增25错，预测差异43行。
新增25由24个malicious→suspicious和1个benign→suspicious组成。seed29减少18个漏报，
虽多7个三分类错误，Score仍更高。因此后续以可用seed29为锚点，46错只作历史实测。
seed30大幅退化说明初始化/批次敏感，不能只报告最优seed掩盖方差。

## 云平台运行

```bash
git fetch origin
git switch release/v4.0-hierarchical
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v4_0_hierarchical
nohup bash scripts/run.sh /root/work > artifacts/v4_0_hierarchical/nohup.log 2>&1 &
echo $! > artifacts/v4_0_hierarchical/train.pid
tail -f artifacts/v4_0_hierarchical/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
