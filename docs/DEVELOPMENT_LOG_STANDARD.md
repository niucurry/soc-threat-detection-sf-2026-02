# SOC 威胁检测开发日志（标准版）

> 标准化日期：2026-09-05
>
> 任务：把每条日志分为 `benign`、`malicious`、`suspicious`
>
> 当前正式候选：`v5.2-exp01-anchor-content-gap2`
> 唯一版本索引：`model_registry.json`

## 1. 先明确版本和证据

旧开发过程把每次实验都顺序叫 V4、V5、V6、V7、V8、V9、V10，造成两个问题：相同基础
算法只改一个输入就升级主版本；H1/H2、A1/B1/C1、seed 和 recovery 又与版本混在一起。
标准版采用以下规则：

- `v1.x`：类别 Embedding + 数值 MLP；v1.1/v1.2 只扩展解析输入；
- `v2.x`：表格神经网络 + 路由 TF-IDF/SGD 专模 + 规则；
- `v3.x`：无模板的内容神经网络；
- `v4.x`：先判威胁、再判威胁子类型的分层神经网络；
- `v5.x`：冻结 v4.0 后添加局部、可回退的冲突残差；
- 同一模型的对照写 `exp01/exp02`，随机重复写 `seed...`，checkpoint 写 `ep...`。

旧名映射为：旧 V4=`v1.1`，旧 V5.1=`v1.2`，旧 V6=`v3.0`，旧 V7=`v4.0`，
旧 V8=`v4.1`，旧 V9=`v5.0`，旧 V10=`v5.1`，旧 V10.1=`v5.2`。旧 V2-F 和
V3-G 分别是 `v2.1`、`v2.2`。

本文严格区分：

1. **完整正式验证**：2,014,052 行官方验证集；
2. **路由内部验证**：只评价 301,333 行正文专模路由，不能冒充完整成绩；
3. **烟雾测试**：只验证代码链路和保护条件；
4. **历史结果但权重丢失**：指标可信，但不能从指标文件恢复模型；
5. **验证集引导实验**：可以帮助定位机制，不能单独证明隐藏集泛化。

## 2. 数据、标签和评估

### 2.1 原始列

三份 Parquet 的原始列完整为：

| 列 | 含义 |
|---|---|
| `event_id` | 行级唯一事件 ID，只用于连接和审计，不作为模型特征 |
| `timestamp` | 事件时间；v1.1 只派生 UTC 小时/星期，原绝对时间不输入 |
| `pipeline` | 采集管道 |
| `src_ip`, `dst_ip` | 源/目的 IP；原值不直接作为高基数类别输入 |
| `src_port` | 源端口 |
| `src_host`, `dst_host` | 源/目的主机；主要派生缺失和长度特征 |
| `username` | 账号；主要派生缺失和长度特征 |
| `message_sanitized` | 脱敏日志正文，是解析和内容建模的核心 |
| `product_name`, `vendor_name` | 产品和厂商 |
| `label_binary` | 名称虽含 binary，实际值为三类标签 |

训练集 2,056,871 行：benign 1,899,723，malicious 111,728，suspicious 45,420。
验证集 2,014,052 行：benign 1,959,573，malicious 14,052，suspicious 40,427。
训练 malicious 占比 5.43%，验证仅 0.70%，存在明显类别先验漂移；所以只追求训练损失或
盲目提高威胁权重会造成大量正常误报。

### 2.2 正式比赛评分

完整评分为六项加权：Threat Binary F1 0.40、Threat Binary Recall 0.25、malicious 和
suspicious Recall 的平均 0.15、Macro-F1 0.10、Soft Label Score 0.05、Balanced
Accuracy 0.05。Soft Label 中精确分类计 1，malicious/suspicious 互错计 0.5，benign/
threat 互错计 0。因此总错误数不是唯一目标：57 个错误的模型可能因 FN 更少而比 50 个错误
模型得分高。每次实验必须同时报告 Score、三分类错误、FP、FN、子类型混淆和 Log Loss。

### 2.3 防泄漏规则

- 类别映射、均值方差、Drain、schema 频次和组合频次只在训练集拟合；
- 验证标签只用于 epoch/阈值选择和最终评价，不参与特征生成；
- `event_id` 只连接数据，不进入模型；原 IP、主机、账号不作为身份类别输入；
- 固定哈希内容编码不拟合验证词表；
- 任何根据验证错误新增的规则或 gap 都标记为验证集引导，必须再做时间外推/OOF 复验。

## 3. v1：同一个表格 MLP，逐步增加解析输入

### 3.1 v1.0：紧凑结构基线

v1.0 的 8 个类别输入完整为：`pipeline`、`product_name`、`product_group`、
`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、
`network_missing_pattern`。

23 个数值输入完整为：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、
`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、
`product_missing`、`message_missing`、`network_present_count`、`message_length`、
`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、
`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、
`message_has_failed`、`message_has_blocked`、`message_starts_angle`、
`message_contains_json`。

每个类别只用训练集映射整数并做 Embedding，未知值为 0；数值用训练均值/标准差标准化并
截断到 [-12,12]。网络为 Embedding 拼接数值后经过 `256-BN-SiLU-Dropout ->
128-BN-SiLU-Dropout -> 64-SiLU -> 3 logits`，交叉熵训练，默认不做类别重权。

完整验证的正式评分 checkpoint：Score `0.9574032044`、Macro-F1 `0.9126510510`、
Accuracy `0.9951178023`、9,833 错；其中 9,793 条是 benign→malicious。旧按 Macro-F1
保存的 checkpoint 为 9,857 错。两者根因相同：正文被压成长度、缺失模式和五个词，多个
完全不同事件得到相同输入。权重扫描没有改变信息瓶颈，所以下一步补正文语义。

### 3.2 v1.1：确定性解析优先，自由文本才用分组 Drain

处理顺序是：识别 Windows JSON/XML、CEF、ASA、VPC Flow、HTTP 等明确格式；这些格式直接
提取字段。剩余自由文本按 `pipeline + vendor + product + message_format` 分组运行 Drain，
防止不同厂商的词位含义混在一起。IP、UUID、邮箱、URL、MAC、长十六进制和数字先类型化；
Drain 只在训练集建树，验证集只能匹配冻结 cluster。

在 v1.0 上新增 14 个类别输入：`vendor_name`、`parser_group`、`message_format`、
`parser_type`、`template_id`、`semantic_action`、`network_protocol`、`event_code`、
`event_name`、`dst_port_bucket`、`http_method`、`http_status_bucket`、`source_zone`、
`destination_zone`。

新增 18 个数值输入：`utc_hour`、`utc_weekday`、`src_port_from_message`、
`dst_port_number`、`dst_port_missing`、`event_code_present`、`semantic_action_present`、
`network_protocol_present`、`semantic_field_count`、`parse_success`、
`template_seen_train`、`template_frequency_log1p`、`template_wildcard_count`、
`message_token_count`、`is_auth_failure`、`is_network_denied`、`is_process_creation`、
`is_privileged_logon`。总输入为 22 类别 + 41 数值，网络仍是 v1.0 MLP。

全量拟合：472,979 行进入 Drain，1,583,892 行直接解析，16 个组、692 个 Drain cluster、
152 个直接模板。完整验证 Score `0.9990226566`、Macro-F1 `0.9990707818`、76 错、0 个
benign→threat。进步证明 action、protocol、event_code 等语义有用。但验证约 953,382 行是
训练未见模板，且 `template_id` 很容易成为标签身份查表。下一步检验更深结构解析是否改善
未见格式，而不是继续盲调 Drain 阈值。

### 3.3 v1.2：深层结构解析实验为什么反而下降

JSON 递归展开对象/数组；Windows XML 先严格解析，失败时有界容错；CEF 分头部和扩展键值；
VPC Flow 按列；ASA/Linux/普通 syslog 保留分组 Drain。

在 v1.1 上新增 14 个类别输入：`structured_parser`、`payload_parse_status`、`schema_id`、
`semantic_template_id`、`event_category_v5`、`event_type_v5`、`event_action_v5`、
`event_outcome_v5`、`event_reason_v5`、`authentication_factor`、`service_name_v5`、
`application_name_v5`、`rule_name_v5`、`threat_category_v5`。

新增 17 个数值输入：`structured_field_count`、`security_field_count`、
`payload_parse_success`、`payload_parse_error`、`schema_seen_train`、
`schema_frequency_log1p`、`semantic_template_seen_train`、
`semantic_template_frequency_log1p`、`source_ip_in_message`、
`destination_ip_in_message`、`event_severity_number`、`malware_present`、
`detection_present`、`authentication_present`、`rule_name_present`、
`user_present_in_payload`、`process_present_in_payload`。总输入为 36 类别 + 58 数值。
`*_v5` 只是旧存储兼容列名。

拟合得到 677 Drain cluster、3,811 直接模板、3,547 schema、3,935 语义模板。云关机发生在
特征写入约 140 万行处；恢复时重跑同一命令，完整分片复用，临时文件原子替换，不删除原始
Parquet。正式结果 Score `0.9984380474`、Macro-F1 `0.9980348900`、142 错，比 v1.1 多
66 错。细粒度组合使 epoch 波动并鼓励按 ID 查表；结构化解析有价值，但不应重新组合为
数千个身份特征。所以下一算法族直接学习内容并禁止所有模板/schema ID。

## 4. v2：双模型混合路线及其边界

v2.0 保留 v1.0 全量三分类基础预测，只把 `pipeline=syslog AND product_name 为空` 路由给
第二模型。该路由训练 342,820 行、验证 301,333 行，仅含 benign/malicious。第二模型使用
全文小写 word 1/2-gram TF-IDF：`min_df=2`、`max_df=0.9999`、最多 200,000 维、
sublinear TF；接 L2、averaged、log-loss SGD 二分类器。明确 `REJECT OK`、Windows 4625、
deny/drop、block-url 等语义由可审计规则覆盖。

v2.1 不改变架构，只让基础 checkpoint 和正文阈值按完整比赛 Score 选择。保守版完整验证
Score `0.9998902649`、Macro-F1 `0.9999579178`、10 个 suspicious→benign、0 FP；调优版
用 Symantec DLP 和 Duo `invalid_passcode+auth_failure` 覆盖这 10 行，达到 1.0。

v2.2 做家族隔离、字符/词符、文本归一化、概率间隙和规则纯度实验。正式 hard result 没变。
这条路线工程效果极强，但满分层利用了同一验证集错误，且模型是“神经网络 + 第二文本模型
+ 规则”，不能回答“单神经网络有没有学会内容”。因此后续主线不用它掩盖基础模型问题，
而把保守 10 错结果留作强基线。

## 5. v3.0：删除模板 ID，直接学习内容

v3.0 禁止 `template_id`、`schema_id`、`semantic_template_id`、Drain cluster、完整消息 ID。
正文屏蔽时间戳、UUID、邮箱、IPv4/IPv6、USER/ORG/CRED/HOST 脱敏实体、长 hex 和长数字，
保留安全词。`raw_token_ids` 完整由归一化单词、相邻 bigram、每个词首/中/尾抽样字符
3/4/5-gram 组成；固定 CRC32 哈希到 65,536 桶，每行最多 96 token，不拟合验证词表。

`field_token_ids` 还加入全部上下文：format、action、protocol、event_code、event_name、
HTTP method、destination port bucket，九类安全信号 malware/potentially_harmful/threat/
authentication/duo_push/no_response/invalid_passcode/powershell/process，VPC log status，以及
从 JSON/XML/键值文本抽出的字段名。

Content encoder 对 token Embedding 做 mean/max pooling并输出 128 维。三组完整结果：

| 实验 | 输入 | Score | 错误 | Macro-F1 | malicious recall |
|---|---|---:|---:|---:|---:|
| v3.0-exp01 | 仅 raw content | 0.9034133234 | 7,567 | 0.8754567628 | 0.5037005408 |
| v3.0-exp02 | v1.0 结构 + raw content | 0.9762209763 | 2,618 | 0.9565217549 | 0.8145459721 |
| v3.0-exp03 | v1.0 结构 + field content | 0.9113565659 | 5,384 | 0.9206922641 | 0.6175633362 |

exp03 说明把字段标记和正文挤进同一短序列会伤害信息覆盖。exp02 的 2,618 错中，2,330 条
是 malicious→suspicious，且都是 VPC `REJECT OK`：训练里该格式来自 AWS 且标 suspicious，
验证新增产品缺失但标 malicious。模型已经识别 threat，只是把“是否威胁”和“威胁业务
子类型”混进同一三分类头。下一版必须结构性拆任务。

## 6. v4：分层模型把 2,618 错降到 46

### 6.1 v4.0 模型定义

Metadata tower 输入 v1.0 全部 8 类别 + `vendor_name`，以及 v1.0 全部 23 数值。Semantic
tower 输入 `content_family`、`content_action`、`content_protocol`、`content_event_code` 四类别，
以及 `content_has_threat`、`content_has_authentication`、
`content_has_potentially_harmful`、`raw_token_count` 四数值。Content tower 输入 96 个
`raw_token_ids`。三塔输出 128/64/128 维。

Threat head 判断 benign/threat，并有 metadata/content 辅助头。Subtype head 先由 metadata
判断 malicious/suspicious，再叠加 semantic/content 的零初始化 residual。概率严格为
`P(malicious)=P(threat)*P(malicious|threat)`，suspicious 同理。损失为 threat CE +
0.75 subtype CE + 0.15 metadata-threat CE + 0.25 content-threat CE +
0.35 metadata-subtype CE。

exp01 不门控；exp02 对训练输入组合 `product_name+content_family+content_action` 计数，使用
`count/(count+32)` 门控 subtype residual。两者 46 个 hard errors 完全相同、Score 均为
`0.9993140394`；exp02 Log Loss `0.0002620398` 优于 exp01 `0.0003264483`。因此 2,572 条
改进来自分层任务，而非 novelty gate。阈值从 0.5 降至 0.2 不改变 46 错，降至 0.02 仅修
2 FN 却新增 97 FP，排除全局阈值方案。

原 46 错 checkpoint 因云环境清理丢失。固定旧 v3.0 特征重训四个 seed 后，seed29 是最高
Score 可用锚点：`0.9993387402`、57 个三分类错、FP=1、FN=32、subtype confusion=24、
Log Loss `0.0002222813`。原结果只作历史参考；所有 V5 改进都与 seed29 epoch-0 逐行比较。

### 6.2 v4.1 为什么没有替换 v4.0

四视图把日志分成 head/middle/tail/key_value，每视图 64 token。key_value 优先字段全集为：
action、act、status、result、reason、outcome、decision、event、eventid、event_id、eventcode、
event_code、process、processname、process_name、command、commandline、command_line、protocol、
severity、category、operation。

标准多视图 208 错/Score `0.9967185764`；raw+证据保持 56 错/`0.9992727909`；
multiview+证据保持 66 错/`0.9992315501`，均未超过 v4.0。全量替换内容塔破坏已有表示；
证据损失恢复退化但产生 FP。结论：保留可靠表示，新证据只能做初始为零、随时回退的残差。

## 7. v5：冻结锚点，只修正有证据的局部冲突

残差可信度输入完整为：冻结 metadata 向量 128、semantic 向量 64、raw content 向量 128，
以及 metadata margin、content margin、novelty gate、`log1p(combo_count)` 四个标量。最终
anchor margin 故意不输入，防止可信度头复制锚点。多视图组额外输入 128 维视图向量；它的
共享 encoder 从锚点复制初始化后仍会训练，原锚点才是冻结的。

候选必须满足 `anchor_margin<0 AND evidence_margin>0`。修正为
`max(0,tanh(trust))*clamp(evidence_margin-anchor_margin,0,max_gap)`，只向 threat 方向；候选外
严格不变，subtype 不变，最后一层全零保证 epoch 0 等于锚点，训练不提升则自动回退。

v5.0 使用 metadata evidence。1,011 个候选全部是真 benign，32 个 FN 一个都不在候选，
所以两个实验都回退 epoch 0，57 错不变。这否定了“放大 metadata 可救漏报”。

错误拆解发现 32 FN 中有 21 个 content probability 0.91 以上，却被 metadata 压回 benign；
另 11 个所有分支都偏 benign；24 个额外错误已经判 threat，只是 malicious→suspicious。
因此 v5.1 反向使用 content evidence。gap=24 能修 21 FN，但 exp01 新增 42 个 FP，最终
43 FP/11 FN/78 错/Score `0.9994966836`；exp02 多视图为 63 FP/11 FN/98 错/
`0.9994142658`。方向正确，允许的残差幅度不合理。

v5.2 把 gap 限为 2，即最多把 threat odds 乘 `exp(2)≈7.39`。这条通用“局部可信域”使
anchor threat 概率远低于约 0.119 的高置信 benign 无法被单个分支直接翻转，不包含 Falcon
厂商规则。

正式默认 exp01：Score `0.9996698618`、36 错、FP=1、FN=11、subtype confusion=24、
相对锚点修正 21、新增 0、Log Loss `0.0002189643`。修复 20 条产品/厂商缺失 malicious
JSON 和 1 条 Cisco Duo suspicious JSON。exp02 hard decisions 完全相同，Log Loss
`0.0002179833`，但特征和网络更复杂，所以不作为默认。

v5.2 是当前验证集上相对可复现锚点的严格改善，但 gap=2 是查看同一验证错误后提出的消融，
还不是独立泛化证据。停止继续扫 gap；下一实验先做训练集时间外推/OOF 校准，再只对已判
threat 的 24 个 subtype confusion 设计独立残差。

## 8. 四个真实事件如何经过各版处理

### 8.1 Cisco ASA：字段可直接解析，不应让 Drain 猜

训练事件 `EVT-0000113120`，标签 suspicious：

```text
<164>Jul 26 USER-9546 05:59:41: USER-0010-0324 Deny udp src dmz-2:10.202.238.40/46422 dst outside:100.64.54.208/53 by ORG-1738-group "USER-6542" [0x0, 0x0]
```

实际解析得到 `message_format=asa`、`action=deny`、`protocol=udp`、源端口 46422、目的端口
53、source zone `dmz-2`、destination zone `outside`、`is_network_denied=1`。IP 和端口被
类型化用于稳定表示，但端口 53 的桶、deny 和 udp 保留。v1.1 把这些显式字段和模板身份送入
MLP；v3.0/v4.0 则让内容 token 学 `deny udp src ... dst ...`，避免只依赖 template_id。

### 8.2 AWS VPC：同样的威胁动作不等于同样的子类型

训练事件 `EVT-0000600002`，标签 suspicious：

```text
2 100000013063 ORG-1504 100.64.0.237 10.182.224.117 48165 39878 6 1 40 1721992874 1721CRED-2CRED-3023300 REJECT OK
```

实际识别 `vpc_flow`、action `reject`、源/目的端口 48165/39878、网络拒绝标志 1；结尾 OK
是日志记录状态，不代表网络允许。v3.0 单头在验证新数据源上把大量同样 `REJECT OK` 的
malicious 判为 suspicious；v4.0 先正确学习 reject=threat，再由 product/metadata 判子类型，
这正是分层模型大幅改进的原因。

### 8.3 Windows 4625：事件码和正文必须同时保留

训练事件 `EVT-0000134221`，标签 suspicious，是一个长 Windows JSON。下列是原文中未改写的
三个真实片段；由于数据已脱敏，winlog 内的事件 ID 键名本身也带脱敏前缀：

```json
"code":"4625","outcome":"failure"
"USER-0010-56507_id":"4625","keywords":["Audit Failure"]
"Failure Reason":"Unknown user name or bad password."
```

同一原文还明确写有 `An account failed to log on`、`Unknown user name or bad password`、
Kerberos 和源地址。实际解析得到 `windows_json`、event code 4625、event name
`logon_failure`、action `fail`、`is_auth_failure=1`。v1.1 可用事件码；v3.0 的 raw token
不会只剩事件码，它还保留 failed/logon/password 等内容和字符片段；v4.0 把事件码语义塔和
原文内容塔分开，避免二者争抢同一 96-token 序列。

### 8.4 Cisco Duo：结构字段可能抽取失败，内容仍能提供证据

训练事件 `EVT-0000441159`，标签 suspicious，原文包含真实片段：

```text
tags=[no_response] ... "event_type":"authentication" ... "factor":"duo_push" ... "reason":"no_response","result":"denied" ... messageType=auth_failure
```

由于字段名也被脱敏扰动，粗结构解析未稳定得到 action，但内容编码仍保留 no_response、
authentication、duo_push、denied、auth_failure。seed29 锚点的一条同类验证事件正是
content 强报警、最终融合判 benign；v5.2 只在锚点边界附近允许内容分支救援，因此修复它，
同时没有用 `product_name=Duo` 写硬规则。

## 9. 当前结论和下一阶段

已经由实验支持的结论：Drain/确定性解析能显著增强结构基线；把过细 template/schema ID
当特征会形成查表风险；固定哈希正文能学习威胁含义，但三分类单头会混淆“威胁检测”和
“业务子类型”；分层任务是单网络最大收益来源；整塔多视图替换不稳定；冻结锚点加局部可信
域能在不新增错误的情况下修复 content 与融合冲突。

尚未证明：v5.2 在未来时间或隐藏格式上仍保持 21/0 的修正精度；11 个全分支低分 FN 能否
仅靠新内容表示修复；24 个 subtype 错是否可在不动 threat 边界时解决。

下一阶段固定 v5.2，不再读取官方验证错误调超参。把训练数据按时间排序，在每个类别内部用
最晚一段构造 temporal holdout；同一日志家族/高度近似事件应整组放在一侧，防止重复泄漏。
同时做训练集 OOF 预测来训练/校准 residual trust。只有在时间外推和 OOF 都不劣于锚点，且
独立检查 FP/FN/subtype 后，才把新方法升级为 v5.3；只换 seed 或 epoch 仍写实验后缀。

## 10. 代码与产物治理

正式入口只有 11 个版本化 shell 脚本，Python 文件按功能命名，不再带历史 V4/V5/V8/V9。
三个重复的结果比较程序已合并为 `src/compare_experiments.py`；已确认无人调用的旧 CSV/SFT
特征恢复代码和无效别名已删除。CatBoost、规则基线和历史比赛材料没有静默删除，因为它们是
结果来源；在 `b0.x` 标为归档基线。历史 Git 提交和旧分支不改写，标准分支映射见
`docs/VERSIONING.md`。

当前 v5.2 默认入口只跑 exp01，必须同时保存 `model.pt`、`metrics.json`、
`valid_predictions.parquet`、预处理器/manifest 和 SHA-256。云环境若仍有旧 V6/V7/V8 目录，
用 `scripts/link_legacy_artifacts.sh` 建只读符号链接，不复制、不删除权重；随后设置
`V5_REUSE_CONTENT_FILES=1` 即可复用旧内容 Parquet。只有模型、指标和逐行预测同时存在，
才把一次训练记为完整。
