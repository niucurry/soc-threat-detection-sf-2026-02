# V1：表格 Embedding + MLP 算法族

V1 只有一个基础网络：离散类别分别做 Embedding，与标准化数值拼接，再经过
`256 -> 128 -> 64 -> 3` 的 MLP。v1.1 和 v1.2 只改变输入，不改变这个基础算法，因此
不能再叫两个新的主版本。

## v1.0：紧凑结构基线

类别输入共 8 个：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、
`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入共 23 个：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、
`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、
`product_missing`、`message_missing`、`network_present_count`、`message_length`、
`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、
`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、
`message_has_failed`、`message_has_blocked`、`message_starts_angle`、
`message_contains_json`。

训练集拟合类别映射和数值均值/标准差；验证集未知类别编码为 0。默认不用类别重权。
完整验证最佳正式评分 checkpoint：Score `0.9574032044`，Macro-F1 `0.9126510510`，
9,833 个错误。根因是正文被压缩为长度、缺失模式和少量关键词，9,793 条正常日志被判为
恶意。下一步不是继续调权重，而是补充正文中的事件语义。

入口：`scripts/run_cloud_v1_0_tabular.sh`。

## v1.1：分格式解析 + 分组 Drain

在 v1.0 的 31 个输入上增加 14 个类别输入：`vendor_name`、`parser_group`、
`message_format`、`parser_type`、`template_id`、`semantic_action`、`network_protocol`、
`event_code`、`event_name`、`dst_port_bucket`、`http_method`、`http_status_bucket`、
`source_zone`、`destination_zone`。

增加 18 个数值输入：`utc_hour`、`utc_weekday`、`src_port_from_message`、
`dst_port_number`、`dst_port_missing`、`event_code_present`、`semantic_action_present`、
`network_protocol_present`、`semantic_field_count`、`parse_success`、
`template_seen_train`、`template_frequency_log1p`、`template_wildcard_count`、
`message_token_count`、`is_auth_failure`、`is_network_denied`、`is_process_creation`、
`is_privileged_logon`。总计 22 个类别、41 个数值输入。

JSON/XML/CEF/ASA/VPC/HTTP 等明确格式由确定性解析器处理；只有不明确的自由文本，才按
`pipeline + vendor + product + format` 分组使用 Drain。Drain 只在训练集拟合，验证集只能
匹配冻结模板。

完整验证结果：Score `0.9990226566`，Macro-F1 `0.9990707818`，76 个错误，正常误报为
威胁为 0。它证明解析有效，但验证中约 95 万行属于训练未见模板，模板编号容易成为高基数
“身份标签”。下一步试验深层结构化解析，同时检验更细模板是否真的泛化。

入口：`scripts/run_cloud_v1_1_drain.sh`。

## v1.2：深层结构解析、schema 与语义模板

在 v1.1 的 63 个输入上增加 14 个类别输入：`structured_parser`、
`payload_parse_status`、`schema_id`、`semantic_template_id`、`event_category_v5`、
`event_type_v5`、`event_action_v5`、`event_outcome_v5`、`event_reason_v5`、
`authentication_factor`、`service_name_v5`、`application_name_v5`、`rule_name_v5`、
`threat_category_v5`。

增加 17 个数值输入：`structured_field_count`、`security_field_count`、
`payload_parse_success`、`payload_parse_error`、`schema_seen_train`、
`schema_frequency_log1p`、`semantic_template_seen_train`、
`semantic_template_frequency_log1p`、`source_ip_in_message`、
`destination_ip_in_message`、`event_severity_number`、`malware_present`、
`detection_present`、`authentication_present`、`rule_name_present`、
`user_present_in_payload`、`process_present_in_payload`。总计 36 个类别、58 个数值输入。

完整验证结果：Score `0.9984380474`，Macro-F1 `0.9980348900`，142 个错误。更细的 schema、
直接模板和语义模板让 epoch 波动加大，并比 v1.1 多 66 个错误。结论不是“结构解析没用”，
而是“把结构重新组合成数千个高基数 ID”会鼓励查表。后续 v3.0 删除模板/schema/cluster ID，
改为直接学习归一化内容。

入口：`scripts/run_cloud_v1_2_structured.sh`。字段名中的 `_v5` 为旧 Parquet/checkpoint
兼容字段，不代表标准模型版本。
