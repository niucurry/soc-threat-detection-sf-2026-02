# v1.1：分组Drain与确定性解析

本分支：`release/v1.1-drain`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v1_1_drain.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

明确格式先解析字段，剩余自由文本按pipeline/vendor/product/format分组Drain，只在训练建模板，验证冻结匹配。新增信息进入同一个MLP。

## 完整输入特征

### 类别输入（22个）

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
| `parser_group` | pipeline、vendor、product、识别格式的分组键。 |
| `message_format` | 识别出的载荷格式，如asa、vpc_flow、windows_json、json或syslog_text。 |
| `parser_type` | 模板路径类别：确定性结构/Drain/未匹配等解析路径。 |
| `template_id` | 分组和模板文本的稳定哈希类别；训练未见值走未知类别。 |
| `semantic_action` | 粗动作归一化，如deny、allow、fail、block、reject或缺失。 |
| `network_protocol` | 正文明确协议字段或协议词；如tcp/udp/icmp。 |
| `event_code` | 事件码字符串；缺失独立编码。 |
| `event_name` | 识别事件码对应的语义名称；未覆盖时保留缺失。 |
| `dst_port_bucket` | 从正文提取目的端口后分桶；缺失独立处理。 |
| `http_method` | 识别HTTP方法。 |
| `http_status_bucket` | HTTP状态码分桶。 |
| `source_zone` | 防火墙源安全区。 |
| `destination_zone` | 防火墙目的安全区。 |

### 数值输入（41个）

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
| `utc_hour` | timestamp按UTC转小时0–23。 |
| `utc_weekday` | timestamp按UTC转星期，周日为0。 |
| `src_port_from_message` | 从正文重新提取的源端口；不能替代原始src_port_number。 |
| `dst_port_number` | 正文中的目的端口数值，未提取为-1。 |
| `dst_port_missing` | dst_port对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `event_code_present` | 是否提取到事件码。 |
| `semantic_action_present` | 是否提取到动作。 |
| `network_protocol_present` | 是否提取到协议。 |
| `semantic_field_count` | 解析器识别出的语义字段数量。 |
| `parse_success` | 基础解析成功标志；不是日志安全标签。 |
| `template_seen_train` | 模板是否在训练模板状态中出现。 |
| `template_frequency_log1p` | 训练模板频次的log(1+count)。 |
| `template_wildcard_count` | 模板通配符数量。 |
| `message_token_count` | 基础解析记录的正文词项数量。 |
| `is_auth_failure` | 按解析动作/事件码等识别的认证失败标志。 |
| `is_network_denied` | 网络拒绝/阻断语义标志。 |
| `is_process_creation` | 进程创建事件语义标志。 |
| `is_privileged_logon` | 特殊权限登录事件标志。 |

## 训练、实验结果、缺陷与后续解决方法

训练默认AdamW，lr=0.002，weight_decay=1e-5，batch=8192，num_workers=4，梯度范数裁剪5，seed=20260828，类别权重power=0。v1.x最多20轮/patience4；v2.x基础最多12轮/patience3。

Drain参数：similarity_threshold=0.5，depth=4，max_children=100，
max_clusters_per_group=5000，max_message_chars=2048，max_groups=128。
分组键是pipeline/vendor/product/format。直接解析器负责动作、协议、事件码和端口；
Drain负责自由文本模板聚类，不会自动变成字段抽取器。动态数值被模板占位前，明确语义字段
已先抽出，因而模型仍能看到目的端口桶、deny、udp。模板ID是稳定哈希类别，不是连续编号值。

全量模板拟合4,465.83秒，完整预处理11,834.60秒。472,979行进入Drain，1,583,892行直接解析，
16组、692个Drain cluster、152个直接模板。验证953,382行模板训练未见，其中230,189行
无法匹配冻结Drain cluster。训练集拟合、验证集只匹配，保证验证文本不更新模板统计。

exp01全量结果：Score=0.9990226565848461，Accuracy=0.9999622651252301，
Balanced Accuracy=0.9983519334181904，Macro-F1=0.9990707818403894，
Log Loss=0.001464069263858073，epoch=1。
混淆矩阵：[[1959573,0,0],[42,13986,24],[10,0,40417]]。
错误76=FP0+FN52+子类型互错24，不能把所有错误统称漏报。

还曾做外接正文专模的工程对照：覆盖301,333行、改变66条分类后，剩10错，
Score=0.9998902649047847，Log Loss=0.0017821218043917778；
再启用可疑规则改变10行后，0错，Score=1，Log Loss=0.0017359466100959691。
这不是单MLP成绩，也不是Drain自身带来的全部收益，应保留为独立外接实验。

模板相关性是风险而不是已经证明的作弊：当前没有只去掉template_id、其余条件不变的配对
实验。变量被占位、格式未见和高基数身份都可能影响结果。后续测试深层结构解析能否保留关系。

## 云平台运行

```bash
git fetch origin
git switch release/v1.1-drain
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v1_1_drain
nohup bash scripts/run.sh /root/work > artifacts/v1_1_drain/nohup.log 2>&1 &
echo $! > artifacts/v1_1_drain/train.pid
tail -f artifacts/v1_1_drain/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
