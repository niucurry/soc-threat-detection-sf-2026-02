# 完整模型输入字典

以下列表从训练代码实际声明生成；每行均说明字段含义。输出Parquet中的审计列不自动成为模型输入。
原始event_id、label_binary、完整时间、原始IP/主机/账号字符串不是神经网络身份特征。

## v1.0及v2.x结构基础输入

### 类别字段（8个）

| 字段 | 含义 |
|---|---|
| `pipeline` | 采集管道；空串/NULL归为缺失类别。 |
| `product_name` | 产品名称；不直接决定威胁标签。 |
| `product_group` | 产品粗组：missing、asa、aws_vpc、other_suspicious_products（Precinct/Falcon）、other。 |
| `src_ip_kind` | 源地址外形类别：missing、ipv4_shape、host_token、other；仅按当前正则判外形。 |
| `port_bucket` | 源端口桶：missing、0–1023、1024–49151、49152以上；基于可解析整数。 |
| `message_length_bucket` | 正文长度桶：missing、1–120、121–180、181–300、301–1000、1001以上。 |
| `structure_combo` | pipeline、product_group、message_length_bucket和是否含deny拼接的类别。 |
| `network_missing_pattern` | 源IP/目的IP/源端口三个缺失位按顺序拼接，如111。 |

### 数值字段（23个）

| 字段 | 含义 |
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

## v1.1完整输入

### 类别字段（22个）

| 字段 | 含义 |
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

### 数值字段（41个）

| 字段 | 含义 |
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

## v1.2完整输入

### 类别字段（36个）

| 字段 | 含义 |
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

### 数值字段（58个）

| 字段 | 含义 |
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

## v4.x/v5.x元数据塔完整输入

### 类别字段（9个）

| 字段 | 含义 |
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

### 数值字段（23个）

| 字段 | 含义 |
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

## v4.x/v5.x语义塔完整输入

### 类别字段（4个）

| 字段 | 含义 |
|---|---|
| `content_family` | 内容解析家族。 |
| `content_action` | 内容粗动作。 |
| `content_protocol` | 内容协议。 |
| `content_event_code` | 内容事件码。 |

### 数值字段（4个）

| 字段 | 含义 |
|---|---|
| `content_has_threat` | 内容安全词组的威胁信号；不是标签。 |
| `content_has_authentication` | 认证关键词信号。 |
| `content_has_potentially_harmful` | potentially harmful相关词组信号。 |
| `raw_token_count` | raw序列非padding数量，上限96。 |

## 内容序列与残差标量

v3.0-exp01只输入`raw_token_ids`。exp02输入上述v1.0全部8类别/23数值加`raw_token_ids`；exp03替换为`field_token_ids`。
v4.0用元数据塔9类别/23数值、语义塔4类别/4数值及`raw_token_ids`；训练频次键为`product_name`、`content_family`、`content_action`。
v4.1-exp01/exp03改用`multiview_token_ids`，exp02仍用raw。四视图顺序head、middle、tail、key_value，每视图64。
v5.x冻结v4.0三塔与分类头，可信度输入为metadata128、semantic64、content128、metadata_margin、content_margin、novelty_gate、log1p(combo_count)，共324维；附加四视图实验再加128维，共452维。
anchor_margin仅用于候选与gap计算，不作为trust网络输入。全部独立输入如上，没有省略的用户ID或隐藏标签输入。
