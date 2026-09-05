# v4.1：多视图与证据保持消融

本分支：`experiment/v4.1-multiview`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v4_1_multiview.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

沿用v4.0分层网络，分别测试原内容/四视图内容与是否添加证据保持损失。四视图共享编码器，每视图64位，共256位；其他输入和子类型门控保留。key_value视图优先字段全集：action、act、status、result、reason、outcome、decision、event、eventid、event_id、eventcode、event_code、process、processname、process_name、command、commandline、command_line、protocol、severity、category、operation。

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

分层训练AdamW lr=0.0007，weight_decay=1e-5，batch=2048，valid_batch=4096，最多10轮，patience4，seed=20260828，threat权重power0.25、subtype权重power0，token_dropout0.05、category_dropout0.02。总损失=threat CE+0.75 subtype CE+0.15 metadata-threat CE+0.25 content-threat CE+0.35 metadata-subtype CE；子类型两项只在真实威胁行计算。

四视图head/middle/tail/key_value，每组64 token，总256；每区域最多4096字符。
共享内容encoder，四个可学习位置向量，再拼接→256→LayerNorm→SiLU→Dropout0.15→
128→LayerNorm→SiLU。键值视图产生key、value、key=value三个符号，优先安全字段，
其余在全文首中尾选取。固定哈希不从验证拟合词表。

证据损失仅在真实threat训练行计算：
max(0,max(0,stop_gradient(max(metadata_margin,content_margin))-0.5)-final_margin)。
附加权重0.20，其余沿用v4.0。冻结目标梯度避免辅助分支通过降低自己输出来逃避约束。
这是训练损失，不是推理时“任一分支报警就判威胁”的硬规则。

| 实验 | 输入/证据权重 | Score | Macro-F1 | 错误 | FP | FN | Log Loss |
|---|---|---:|---:|---:|---:|---:|---:|
| exp01 | 四视图/0 | 0.9967185764373843 | 0.997575995559513 | 208 | 0 | 208 | 0.000923406826650498 |
| exp02 | raw/0.20 | 0.9992727909041258 | 0.9994852508215298 | 56 | 10 | 46 | 0.0003202954079632288 |
| exp03 | 四视图/0.20 | 0.9992315501380292 | 0.9994431890113717 | 66 | 20 | 46 | 0.00041433731422238284 |

三组最佳epoch均1，subtype互错均0。exp01的198恶意FN+10可疑FN，共208；
相对v4.0新增162个恶意FN，分布110个syslog_text/deny、48个JSON/block、4个WindowsJSON/deny。
48个block中32个已经有content_has_threat=1，故不能简单认为输入完全没有安全信号。

exp02未修原46个FN，新增10个FP全为Crowdstrike Falcon/json/action缺失，
真benign判suspicious，组合频次10–99。exp03修回exp01新增162个FN，但保留原46个，
新增20个FP：Falcon9、ASA6、VMWare VCenter/fail4、Duo/success1，全判suspicious。
count门控只限制subtype残差，因此不能阻止这些威胁门的FP/FN。

完整覆盖统计：head平均32.50/64，middle32.44/64，tail32.45/64，key_value23.94/64；
四组都满64共71,378行。困难JSON大多三段满64，键值58–64。容量被用满不等于决定字段已进入。
以下为已回传分支概率均值：

| 实验/错误家族 | final threat | metadata threat | content threat |
|---|---:|---:|---:|
| exp02、28个JSON认证FN | 0.001522 | 0.005295 | 0.000264 |
| exp02、8个Symantec success FN | 0.000010 | 0.000095 | 0.000014 |
| exp02、6个Windows deny FN | 0.003532 | 0.971453 | 0.000015 |
| exp03、28个JSON认证FN | 0.026100 | 0.005082 | 0.000220 |
| exp03、6个Windows deny FN | 0.057695 | 0.972373 | 0.000032 |
| exp03、Falcon FP | 0.661802 | 0.064806 | 0.003010 |
| exp03、ASA FP | 0.749334 | 0.067490 | 0.002770 |
| exp03、VCenter FP | 0.807723 | 0.003570 | 0.017447 |
| exp03、Duo FP | 0.789687 | 0.005055 | 0.000278 |

新增FP的两个辅助头都低，final却高，说明是融合交互和训练边界变化，不能说成简单“照搬高分分支”。
exp03有290个final benign/content threat冲突，其中真threat为0；这一模型上不能用content硬覆盖。
不同锚点的分支概率会改变，后续恢复模型必须重新审计冲突，不能套用此处290/0结论。
三组Log Loss也均高于v4.0，故没有替换主模型。下一步冻结可靠锚点，仅允许局部残差。

## 云平台运行

```bash
git fetch origin
git switch experiment/v4.1-multiview
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v4_1_multiview
nohup bash scripts/run.sh /root/work > artifacts/v4_1_multiview/nohup.log 2>&1 &
echo $! > artifacts/v4_1_multiview/train.pid
tail -f artifacts/v4_1_multiview/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
