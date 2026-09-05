# v1.2：深层结构解析与schema特征

本分支：`experiment/v1.2-structured`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v1_2_structured.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

进一步解析嵌套JSON/XML、CEF、固定列载荷，增加schema及语义身份和字段。这是输入变化，神经网络仍为同一结构。

## 完整输入特征

### 类别输入（36个）

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
| `structured_parser` | 深层载荷解析器类别。 |
| `payload_parse_status` | success、partial、failed、not_applicable或blank。 |
| `schema_id` | 结构字段路径集合稳定哈希。 |
| `semantic_template_id` | 格式/schema/事件语义组合稳定哈希。 |
| `event_category` | 深层字段推导的事件类别。 |
| `event_type` | 深层字段推导的事件类型。 |
| `event_action` | 结合基础动作和深层字段的动作。 |
| `event_outcome` | 结果，如success/failure/unknown。 |
| `event_reason` | 结果原因字段。 |
| `authentication_factor` | 认证因子，如duo_push。 |
| `service_name` | 载荷服务名称。 |
| `application_name` | 载荷应用名称。 |
| `rule_name` | 载荷规则名称。 |
| `threat_category` | 载荷威胁类别。 |

### 数值输入（58个）

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
| `structured_field_count` | 展开后结构字段数量。 |
| `security_field_count` | 被认为有安全语义的字段数量。 |
| `payload_parse_success` | 深层载荷解析成功标志。 |
| `payload_parse_error` | 深层载荷解析错误标志。 |
| `schema_seen_train` | schema是否训练已见。 |
| `schema_frequency_log1p` | 训练schema频次log1p。 |
| `semantic_template_seen_train` | 语义模板是否训练已见。 |
| `semantic_template_frequency_log1p` | 训练语义模板频次log1p。 |
| `source_ip_in_message` | 载荷识别的源IP是否存在。 |
| `destination_ip_in_message` | 载荷识别的目的IP是否存在。 |
| `event_severity_number` | 深层字段提取的严重度数值。 |
| `malware_present` | 载荷恶意软件语义存在标志。 |
| `detection_present` | 检测语义存在标志。 |
| `authentication_present` | 认证语义存在标志。 |
| `rule_name_present` | 是否识别规则名。 |
| `user_present_in_payload` | 载荷用户相关字段是否存在。 |
| `process_present_in_payload` | 载荷进程相关字段是否存在。 |

## 训练、实验结果、缺陷与后续解决方法

训练默认AdamW，lr=0.002，weight_decay=1e-5，batch=8192，num_workers=4，梯度范数裁剪5，seed=20260828，类别权重power=0。v1.x最多20轮/patience4；v2.x基础最多12轮/patience3。

JSON递归展开对象路径，数组记录长度并读取前32个元素，递归深度上限8。
Windows XML严格解析优先，失败后有界扫描；CEF拆头与键值扩展；VPC按固定列。
解析状态区分success、partial、failed、not_applicable、blank。解析失败不表示事件正常。
schema由字段路径集合决定；语义模板再组合格式、schema、事件码、动作与结果。

模板统计：677个Drain cluster、3,811个直接模板、3,547个schema、3,935个语义模板。
拟合9,542.49秒；云关机时已写约140万训练行。模板保存完成不代表最终特征完成；
恢复需检查manifest和完整分片，重跑未完成阶段。nohup只能防终端断开，不能跨机器关机续跑。

正式Score=0.998438047351911，Macro-F1=0.998034890020854，
Accuracy=0.9999294953655615，Balanced Accuracy=0.9967863199823804，
Log Loss=0.0008156300414003951，epoch=4。
矩阵[[1959573,0,0],[48,13920,84],[10,0,40417]]。
142错=FP0+FN58+子类型互错84；比v1.1多6个FN和60个子类型错误。

Log Loss更低而分类更差并不矛盾：它平均所有行的概率损失，大量正确行的概率改善可以
超过少量困难行的损失。更细ID、高基数组合和训练波动是可能原因；未做仅删除schema的
严格消融，不能认定某个特征是唯一根因。下一步正文模型明确不输入模板、schema或消息ID。

## 云平台运行

```bash
git fetch origin
git switch experiment/v1.2-structured
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v1_2_structured
nohup bash scripts/run.sh /root/work > artifacts/v1_2_structured/nohup.log 2>&1 &
echo $! > artifacts/v1_2_structured/train.pid
tail -f artifacts/v1_2_structured/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
