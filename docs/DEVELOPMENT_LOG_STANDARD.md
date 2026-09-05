# SOC 威胁检测开发日志（标准版）

> 标准化日期：2026-09-05
>
> 任务：把每条日志分为 `benign`、`malicious`、`suspicious`
>
> 当前正式候选：`v5.2-exp01-anchor-content-gap2`
> 唯一版本索引：`model_registry.json`

## 1. 研究问题、阅读顺序与证据

本项目研究如何从单条脱敏安全日志中识别 benign、malicious、suspicious。实验有两条路线：
表格模型及双模型混合路线，和直接学习内容的分层神经网络路线。前者验证工程成绩上限，
后者检验内容表示、任务拆分和局部修正是否有效。全文按模型家族组织，同一家族内按次版本
和实验编号说明改动；一次随机种子或阈值对照只记实验，不产生新模型家族。

本文采用以下正式编号：

- `v1.x`：类别 Embedding + 数值 MLP；v1.1/v1.2 只扩展解析输入；
- `v2.x`：表格神经网络 + 路由 TF-IDF/SGD 专模 + 规则；
- `v3.x`：无模板的内容神经网络；
- `v4.x`：先判威胁、再判威胁子类型的分层神经网络；
- `v5.x`：冻结 v4.0 后添加局部、可回退的冲突残差；
- 同一模型的对照写 `exp01/exp02`，随机重复写 `seed...`，checkpoint 写 `ep...`。

本文严格区分：

1. **完整正式验证**：2,014,052 行官方验证集；
2. **路由内部验证**：只评价 301,333 行正文专模路由，不能冒充完整成绩；
3. **烟雾测试**：只验证代码链路和保护条件；
4. **历史结果但权重丢失**：指标可信，但不能从指标文件恢复模型；
5. **验证集引导实验**：可以帮助定位机制，不能单独证明隐藏集泛化。

## 2. 数据、标签和评估

### 2.1 原始列

三个文件并不拥有相同列。训练文件含下列全部 13 列；验证输入含除标签以外的 12 列；
验证答案文件仅含 `event_id` 和 `label_binary`。必须按 ID 连接答案，不能按物理行序拼接。
完整字段定义为：

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


## 3. 模型开发与完整实验

各节包含模型、完整输入、训练参数、实测结果、失败诊断与下一步。未留存的指标明确不填。

### v1.0：结构Embedding与MLP

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

目标是建立紧凑基线，正文只以长度和关键词指示进入，无法完整表达嵌套事件。

类别输入完整8项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

在神经网络前曾准备 CatBoost 候选：MultiClass，350 iterations，depth=8，
learning_rate=0.12，l2_leaf_reg=5，Balanced class weights，random_strength=0.5，
seed=20260828。未留存完整验证结果，因此只作为方法探索记录，不列为有成绩的模型版本。

神经网络采用 AdamW，lr=0.002，weight_decay=1e-5，batch=8192，梯度范数裁剪5，
seed=20260828。类别权重为 [N/(K*n_class)]^power，正式使用power=0，即全部为1。
类别 Embedding 维数 min(24,max(3,round(2*cardinality^0.25)))。三类输出顺序固定为
benign、malicious、suspicious。两个 checkpoint 必须分别记录：

| 实验 | 选择指标 | epoch | Score | 错误 | FP | FN | 子类型互错 |
|---|---|---:|---:|---:|---:|---:|---:|
| exp01 | Macro-F1 | 1 | 0.9572098298 | 9,857 | 9,793 | 40 | 24 |
| exp02 | Competition Score | 3 | 0.9574032044 | 9,833 | 9,793 | 40 | 0 |

exp01混淆矩阵：[[1949780,9793,0],[30,13998,24],[10,0,40417]]；
exp02：[[1949780,9793,0],[30,14022,0],[10,0,40417]]。
exp01 Macro-F1约0.912286，exp02为0.9126510510016193。
exp02云端训练耗时227.10秒；初始记录每epoch约35–37秒，checkpoint为236,108字节。
这些耗时是单次环境测量，不能当作所有机器的保证。

权重对照计划在相同20万行上比较0、0.25、0.50、0.75。仅power=0全量结果可核实，
其他权重未留存完整成绩。训练损失下降但验证分类不动，说明继续增加训练轮次未解决问题。
9,793个正常误报集中在产品缺失syslog路由：该路由训练231,092 benign/111,728 malicious，
验证287,281 benign/14,052 malicious。攻击比例由32.59%变4.66%，原结构表示又把不同正文
压成同样输入，支持“信息不足并伴随先验漂移”的诊断。后续分别尝试解析特征和正文专模。

### v1.1：分组Drain与确定性解析

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

明确格式先解析字段，剩余自由文本按pipeline/vendor/product/format分组Drain，只在训练建模板，验证冻结匹配。新增信息进入同一个MLP。

类别输入完整22项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`、`parser_group`、`message_format`、`parser_type`、`template_id`、`semantic_action`、`network_protocol`、`event_code`、`event_name`、`dst_port_bucket`、`http_method`、`http_status_bucket`、`source_zone`、`destination_zone`。

数值输入完整41项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`、`utc_hour`、`utc_weekday`、`src_port_from_message`、`dst_port_number`、`dst_port_missing`、`event_code_present`、`semantic_action_present`、`network_protocol_present`、`semantic_field_count`、`parse_success`、`template_seen_train`、`template_frequency_log1p`、`template_wildcard_count`、`message_token_count`、`is_auth_failure`、`is_network_denied`、`is_process_creation`、`is_privileged_logon`。

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

### v1.2：深层结构解析与schema特征

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

进一步解析嵌套JSON/XML、CEF、固定列载荷，增加schema及语义身份和字段。这是输入变化，神经网络仍为同一结构。

类别输入完整36项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`、`parser_group`、`message_format`、`parser_type`、`template_id`、`semantic_action`、`network_protocol`、`event_code`、`event_name`、`dst_port_bucket`、`http_method`、`http_status_bucket`、`source_zone`、`destination_zone`、`structured_parser`、`payload_parse_status`、`schema_id`、`semantic_template_id`、`event_category`、`event_type`、`event_action`、`event_outcome`、`event_reason`、`authentication_factor`、`service_name`、`application_name`、`rule_name`、`threat_category`。

数值输入完整58项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`、`utc_hour`、`utc_weekday`、`src_port_from_message`、`dst_port_number`、`dst_port_missing`、`event_code_present`、`semantic_action_present`、`network_protocol_present`、`semantic_field_count`、`parse_success`、`template_seen_train`、`template_frequency_log1p`、`template_wildcard_count`、`message_token_count`、`is_auth_failure`、`is_network_denied`、`is_process_creation`、`is_privileged_logon`、`structured_field_count`、`security_field_count`、`payload_parse_success`、`payload_parse_error`、`schema_seen_train`、`schema_frequency_log1p`、`semantic_template_seen_train`、`semantic_template_frequency_log1p`、`source_ip_in_message`、`destination_ip_in_message`、`event_severity_number`、`malware_present`、`detection_present`、`authentication_present`、`rule_name_present`、`user_present_in_payload`、`process_present_in_payload`。

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

### v2.0：结构基础模型与路由正文专模

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

第一模型负责全量；第二模型在syslog且product_name缺失路由上训练TF-IDF/SGD二分类。路由外保留第一模型，路由内正文预测覆盖，再按实验选择是否启用可疑规则。基础轮次按Macro-F1、正文阈值按路由Macro-F1选，规则使用basic范围。

类别输入完整8项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

路由为pipeline=syslog且product_name为空。TF-IDF完整设置：
lowercase=True，analyzer=word，ngram_range=(1,2)，min_df=2，max_df=0.9999，
max_features=200000，sublinear_tf=True，dtype=float32，
token_pattern为匹配长度至少2的字母数字下划线、点和连字符词项。
SGDClassifier：loss=log_loss，penalty=l2，alpha=1e-6，max_iter=30，
tol=1e-4，average=True，seed=20260828。只在路由训练正文拟合。

| 实验 | 范围 | 阈值 | Macro-F1 | FP | FN | 结论 |
|---|---|---:|---:|---:|---:|---|
| exp01 | 路由301,333行 | 0.5 | 0.759090 | 0 | 8,940 | 新格式召回不足 |
| exp02 | 同一路由 | 约0.048448 | 0.947411 | 2 | 2,570 | 调阈值仍缺格式证据 |
| exp03 | 路由加首轮规则 | 按路由选择 | 0.999159 | 1 | 44 | 缺少明确BLOCKED格式 |
| exp04 | 路由加完整基础规则 | 约0.056902 | 1.0 | 0 | 0 | 8,024个恶意命中规则 |
| exp05 | 全量保守融合 | 专模保存值 | 0.9999579178 | 0 | 10 | 10个suspicious漏报 |
| exp06 | 全量可疑覆盖 | 同上 | 1.0 | 0 | 0 | 验证引导规则，不是泛化证明 |

基础规则完整条件：包含reject ok；紧凑JSON含code=4625；以org-1780 ::: tags=开始；
以org-1780 ::: fqdn=开始且含=blocked；包含空格包围deny；包含,traffic,deny,。
ORG标记是脱敏数据格式条件，并非厂商无关语义。规则只作用于指定路由。
额外可疑覆盖：Symantec Data Loss Prevention产品；或者Duo产品且正文同时含
invalid_passcode和auth_failure。前者训练无同产品支持、验证9行；后者训练2行、验证1行。
不能用这些结果推出“安全设备的拒绝日志全部是威胁”。

融合按event_id检查唯一性、存在性、标签一致性后覆盖。保守全量矩阵
[[1959573,0,0],[0,14052,0],[10,0,40417]]，重算Score=0.9998902649047847；
可疑覆盖后完全对角，Score=1。路由指标不能拿来当全量指标。

### v2.1：混合模型全量评分选择

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

同一双模型与basic规则；基础轮次与正文阈值按全量Competition Score选。它是选择策略变更，不是增加第三个模型。

类别输入完整8项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

架构不变，基础checkpoint改按全量Competition Score选。正文阈值先计算路由外1,712,719行
固定混淆矩阵，再累加每个候选边界的路由TP/FP/FN/TN，最大化全量六项分数；
1e-12同分内偏向预测更多威胁的边界。

本地阈值0.056902363896369934，云端0.05870809219777584；
两者在各自概率下都将路由分对。阈值必须与配套模型一起保存，不能跨模型直接照搬。
云端结构训练227.10秒，正文训练与评价60.57秒，词表200,000。
正文修复基础分类9,823条：9,793 benign→malicious和30 malicious→benign；
再加可疑覆盖共修复9,833条。

| 实验 | 全量Score | Macro-F1 | 错误 | FP | FN | 子类型互错 |
|---|---:|---:|---:|---:|---:|---:|
| exp01保守 | 0.9998902649047847 | 0.9999579178042195 | 10 | 0 | 10 | 0 |
| exp02可疑覆盖 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |

保守结果Threat-F1=0.9999082130924845，Threat-Recall=0.9998164430330954，
两威胁类平均召回=0.9998763202810004，Soft-Score=0.9999950348848987，
Balanced-Accuracy=0.9999175468540002。可疑覆盖版六项均1。
正文阈值和可疑规则依赖同一验证分布，下一步需跨家族检查。

### v2.2：混合模型语义规则与泛化审计

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

保持全量Score选择，expanded规则增加两个精确语义条件；正文仍是TF-IDF/SGD。家族/字符实验独立于正式推理。

类别输入完整8项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

精确原文审计：954,292个验证benign在训练出现且标签一致；malicious和suspicious原文重叠为0。
长hex和数字归一化后，这两类完整模板重叠仍为0；归一化前48字符家族覆盖5,266个验证恶意。
验证209行来自2个未见产品，pipeline无新增枚举。正常原文重复与攻击格式漂移同时存在。

| 实验 | 实测结果 | 决策 |
|---|---|---|
| 模板首折 | 272,972训练/69,848留出；词80.73秒，字符263.83秒；内部均约0.999970 Macro-F1、2错 | 内部近满分未迁移外部 |
| 首折外部 | 词原始Macro-F1约0.75909、恶意召回0.36379；加基础规则约0.85828、召回0.57102 | 字符/词符融合无改善且更慢 |
| 五折家族隔离 | 内部错误0、0、0、2、24；外部原始恶意召回均约36.379%，加规则约57.102% | 当前隔离不足以模拟完整厂商迁移 |
| IP/hex/数字归一化 | 词表200,000→18,669；约38→24秒；外部只多识别2个恶意 | 未替换正式模型 |
| 无标签间隙阈值 | 五折错误4、1、0、38、287；阈值约0.0612、0.0413、0.0400、0.0713、0.1302 | 不稳定，未采用 |
| 五折概率融合 | 均值、中位数、最小、最大、logit均比较 | 未解决不同折压低不同家族 |

排除基础规则后，benign最大恶意概率约0.054065，malicious最小约0.059739，间隔0.005674；
6,028个未覆盖恶意的中位概率约0.21654。高权重有deny/protocol/dport，也有0x0/src/dst、
身份与时间片段；分类全对仍可能对漂移敏感。

| 规则候选 | 训练恶意 | 验证恶意 | benign命中 | 新增覆盖 |
|---|---:|---:|---:|---:|
| TRAFFIC,drop | 0 | 5,826 | 0 | 5,826 |
| THREAT,url且block-url | 0 | 200 | 0 | 200 |
| 协议字段后deny | 0 | 5,820 | 0 | 5,820 |
| 任意,deny, | 0 | 5,838 | 0 | 5,826 |
| decision=blocked | 260 | 0 | 0 | 260 |
| act=deny | 50 | 0 | 0 | 50 |

正式只新增,traffic,drop,和,threat,url,且block-url两条件。总计14,050恶意、0正常命中。
剩2条Windows4672由模型判断，没有硬写4672=恶意：授予特殊权限也可由正常系统账号触发。

本地阈值0.17455598153173923，云端0.1985421571880579，正文耗时59.31秒，
路由Log Loss=0.008998715901508317。

| 实验 | 全量Score | Macro-F1 | 错误 | 全量Log Loss |
|---|---:|---:|---:|---:|
| exp01保守 | 0.9998902649047847 | 0.9999579178042195 | 10 | 0.0014145733444869055 |
| exp02可疑覆盖 | 1.0 | 1.0 | 0 | 0.0013629193716559853 |

规则扩大当前概率间隔，但新规则纯度来自反复使用的验证集，尚无独立实验支持跨格式鲁棒性。
这一模型包含两个分类器与规则，之后的单神经网络路线单独报告。

### v3.0：无模板内容神经网络

每条日志生成96位固定哈希内容序列；可学习64维Embedding经mean/max pooling得到128维，再经MLP输出内容表示。exp01仅raw序列，exp02增加结构塔，exp03用field序列替代raw；三分类融合头与内容辅助头共同训练。field_token_ids的上下文全集为format、action、protocol、event_code、event_name、http_method、dst_port_bucket、VPC log_status、JSON/XML/键值文本字段名，以及malware、potentially_harmful、threat、authentication、duo_push、no_response、invalid_passcode、powershell、process九类安全信号。它们按有限预算取入，不保证全部同时保留。

exp01只用raw_token_ids。下列结构字段用于exp02/exp03，另分别加入raw_token_ids/field_token_ids。

类别输入完整8项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

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

### v4.0：威胁与子类型分层神经网络

元数据、语义、raw内容三塔分别128/64/128维。先预测benign/threat，再用metadata子类型头与semantic/content残差判断malicious/suspicious。概率为P(m)=P(t)P(m|t)、P(s)=P(t)P(s|t)，决策先按threat阈值再取子类型最大值。频次门控count/(count+32)只限制子类型残差。

类别输入完整9项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

语义类别完整4项：`content_family`、`content_action`、`content_protocol`、`content_event_code`。

语义数值完整4项：`content_has_threat`、`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。

内容为raw_token_ids；多视图实验用multiview_token_ids。频次键为product_name、content_family、content_action。

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

### v4.1：多视图与证据保持消融

沿用v4.0分层网络，分别测试原内容/四视图内容与是否添加证据保持损失。四视图共享编码器，每视图64位，共256位；其他输入和子类型门控保留。key_value视图优先字段全集：action、act、status、result、reason、outcome、decision、event、eventid、event_id、eventcode、event_code、process、processname、process_name、command、commandline、command_line、protocol、severity、category、operation。

类别输入完整9项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

语义类别完整4项：`content_family`、`content_action`、`content_protocol`、`content_event_code`。

语义数值完整4项：`content_has_threat`、`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。

内容为raw_token_ids；多视图实验用multiview_token_ids。频次键为product_name、content_family、content_action。

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

### v5.0：冻结锚点的元数据冲突残差

冻结v4.0-exp02-seed20260829全部参数。仅在anchor判benign、metadata判threat时允许非负局部修正；最大logit gap=24。epoch0等于锚点，验证未改善则回退。子类型始终来自锚点。

类别输入完整9项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

语义类别完整4项：`content_family`、`content_action`、`content_protocol`、`content_event_code`。

语义数值完整4项：`content_has_threat`、`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。

内容为raw_token_ids；多视图实验用multiview_token_ids。频次键为product_name、content_family、content_action。

冻结v4.0-exp02-seed20260829。候选是anchor margin<0且metadata margin>0；
margin是threat logit减benign logit。delta=candidate*max(0,tanh(trust_logit))*
clamp(evidence_margin-anchor_margin,0,24)。candidate外逐行不变，子类型不变；
零初始化trust输出层保证epoch0等于锚点。max(0,tanh)限制非负delta，最多靠近证据分支。

可信度输入包括metadata128、semantic64、raw content128以及metadata margin、content margin、
novelty gate、log1p(combo_count)四标量。exp02再加128维四视图向量，其encoder复制锚点初始化
后仍可训练；原锚点才被冻结。最终anchor margin只进入候选和gap计算，不输入trust网络。

训练正例为真实threat且指定证据分支判threat，困难负例从benign按证据margin降序选最多正例2倍。
若只选训练集实际冲突，烟雾测试出现0样本，无法训练，因此改为学习证据可信度；
推理仍严格限定真正冲突。总损失=trust二元CE+最终威胁CE+0.10*锚点KL蒸馏+
0.01*trust_score平方均值。AdamW lr=0.0005，weight_decay=0.0001，batch=512，
scan/valid_batch=4096，epochs≤8，patience=3，梯度裁剪5，seed=20260904。

| 实验 | Score | 错误 | FP | FN | subtype | best_epoch | 改变分类 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp01锚点表示 | 0.9993387401713992 | 57 | 1 | 32 | 24 | 0 | 0 |
| exp02附加四视图 | 0.9993387401713992 | 57 | 1 | 32 | 24 | 0 | 0 |

验证metadata判threat共55,450行，其中真threat54,439、benign1,011；
anchor benign/metadata threat候选恰是1,011个benign，真threat0。
32个FN全部被外层mask排除，故不存在可修正FN；增加视图也无法突破候选边界。
两组自动回退是有效否定结果，不代表训练后提高了模型。

恢复锚点的57错进一步分解：

- 20个产品缺失JSON恶意FN：final平均0.438180、metadata0.003641、content0.947359；
  content范围0.912818–0.961938。
- 1个Duo可疑FN：final0.253372、metadata0.000602、content0.938998。
- 11个全分支偏benign的FN：8个Symantec success、2个无产品WindowsJSON、1个Symantec其他事件。
- 24个malicious→suspicious：WindowsJSON/fail，final0.938884、metadata0.877243、content0.000036。
- 1个Duo正常误报：final0.515301、metadata0.034732、content0.466180。

21个FN存在独立内容证据，下一次版本只换证据源；24个子类型错误和1个已有FP不在向上残差可修范围。

### v5.1：冻结锚点的内容冲突残差

冻结同一个v4.0锚点，候选证据改为content，max_conflict_gap=24。通过训练正例与高分困难正常样本学习可信度，只修正候选内威胁logit，保持子类型和候选外预测。

类别输入完整9项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

语义类别完整4项：`content_family`、`content_action`、`content_protocol`、`content_event_code`。

语义数值完整4项：`content_has_threat`、`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。

内容为raw_token_ids；多视图实验用multiview_token_ids。频次键为product_name、content_family、content_action。

只将证据源改为content，保持gap24、冻结锚点和训练保护。反向冲突在全验证集上的审计：

| content阈值 | 候选 | 真threat | benign | 精度 |
|---:|---:|---:|---:|---:|
| 0.50 | 294 | 21 | 273 | 0.071429 |
| 0.70 | 174 | 21 | 153 | 0.120690 |
| 0.90 | 41 | 21 | 20 | 0.512195 |
| 0.91 | 36 | 21 | 15 | 0.583333 |
| 0.92 | 29 | 17 | 12 | 0.586207 |
| 0.94 | 20 | 16 | 4 | 0.800000 |
| 0.96 | 9 | 8 | 1 | 0.888889 |

高阈值仍有FP且会丢正例，所以正式模型保留0.5宽候选，用训练标签学trust。

| 实验 | Score | 错误 | FP | FN | subtype | epoch | 修复/新增 |
|---|---:|---:|---:|---:|---:|---:|---|
| exp01锚点内容 | 0.9994966836 | 78 | 43 | 11 | 24 | 1 | 21/42 |
| exp02附加四视图 | 0.9994142658 | 98 | 63 | 11 | 24 | 2 | 21/62 |

两组修复同21个FN（20个无产品恶意JSON、1个Duo可疑JSON），新增FP均集中Falcon；
exp01共改63行，正确修复比例21/63=33.33%；exp02改83行，比例25.30%。
exp01 Log Loss约0.0002541，高于锚点0.0002223。Score提高源于比赛更重视威胁召回，
并非所有业务指标均提高。四视图新增20个FP，未选为主候选。

真威胁20条anchor平均0.438、最小0.316，Duo0.253；Falcon新增FP anchor平均0.0135、
content平均0.899。gap24允许可信度高时推翻极高置信benign。下一次版本限制logit最大修正，
不加入厂商名规则。

### v5.2：内容救援与局部幅度限制

冻结同一个v4.0锚点，使用content证据，将max_conflict_gap限制为2。epoch0不改锚点；exp01用冻结表示训练可信度，exp02可增加四视图。默认采用exp01。

类别输入完整9项：`pipeline`、`product_name`、`product_group`、`src_ip_kind`、`port_bucket`、`message_length_bucket`、`structure_combo`、`network_missing_pattern`、`vendor_name`。

数值输入完整23项：`src_port_number`、`src_ip_missing`、`dst_ip_missing`、`src_port_missing`、`src_host_missing`、`dst_host_missing`、`username_missing`、`product_missing`、`message_missing`、`network_present_count`、`message_length`、`src_ip_length`、`dst_ip_length`、`src_host_length`、`dst_host_length`、`username_length`、`message_has_deny`、`message_has_allow`、`message_has_accepted`、`message_has_failed`、`message_has_blocked`、`message_starts_angle`、`message_contains_json`。

语义类别完整4项：`content_family`、`content_action`、`content_protocol`、`content_event_code`。

语义数值完整4项：`content_has_threat`、`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。

内容为raw_token_ids；多视图实验用multiview_token_ids。频次键为product_name、content_family、content_action。

将max_conflict_gap设2，其余方向不变。delta≤2，threat odds最多乘exp(2)≈7.39。
若anchor概率<sigmoid(-2)≈0.1192，则不可能越过0.5；该概率边界针对默认threat阈值0.5。

| 实验 | Score | Macro-F1 | 错误 | FP | FN | subtype | 修复 | 新错 | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp01锚点内容 | 0.9996698618032245 | 0.9995501728 | 36 | 1 | 11 | 24 | 21 | 0 | 0.0002189643 |
| exp02附加四视图 | 0.9996698618032245 | 0.9995501728 | 36 | 1 | 11 | 24 | 21 | 0 | 0.0002179833 |

由已知错误方向复原的全量矩阵为[[1959572,0,1],[2,14026,24],[9,0,40418]]；
该矩阵是计数推导，不冒充新增一次推理。exp02分类完全相同，Macro-F1因此相同。
最佳epoch的完整精确记录未在本节资料中给出，应以各模型metrics.json核实，不填假值。
malicious召回0.9981497，suspicious召回0.9997774，benign召回维持锚点。

候选仍294=21个真威胁+273个benign，只翻转21个真威胁，三分类和威胁二分类fixed均21、
new均0。20个恶意最终平均threat0.8478，Duo0.7149。Falcon组236个候选平均trust0.8266、
delta1.6532，但anchor平均0.00607、最终平均0.03781，无新增FP。
这证明此组实验中幅度限制有效，即使trust仍较高也能保留极低anchor判断。

exp02只将Log Loss降低约9.81e-7，没有多修一行，计算更复杂，默认选exp01。
当前剩36错由1个FP、11个FN、24个子类型混淆构成。单向threat修正不能修复已有FP，
也不会动24个已判threat的子类型；11个共同低分FN需新的证据或上下文实验。
gap2由同一验证错误分布启发，21修复/0新增是开发验证结论，未证明未来数据仍成立。
下一阶段先时间外推和训练集OOF校准，再测试只作用threat内部的subtype残差。

## 4. 真实数据案例


以下四行于2026-09-05从train.parquet按event_id精确读取。input包含实际全部13列，message_sanitized完整保留；parsed为当前parse_log实际输出，未使用拟合模板模型，因此不伪造训练模板ID或频次。
这些样本不是新增训练实验，也不是模型预测。字段被脱敏时，其键名本身可能改变。

### EVT-0000113120 · ASA Firewall

明确格式识别为asa，直接得到deny、udp和目的端口53。可从语义字段建模，Drain不是字段抽取的来源。

完整原始输入：

```json
{
  "event_id": "EVT-0000113120",
  "timestamp": 1721992226.4811196,
  "pipeline": "syslog",
  "src_ip": "10.247.160.227",
  "dst_ip": "198.51.100.98",
  "src_port": "46422",
  "src_host": "10.232.175.9",
  "dst_host": "",
  "username": "",
  "message_sanitized": "<164>Jul 26 USER-9546 05:59:41: USER-0010-0324 Deny udp src dmz-2:10.202.238.40/46422 dst outside:100.64.54.208/53 by ORG-1738-group \"USER-6542\" [0x0, 0x0]\n",
  "product_name": "ASA Firewall",
  "vendor_name": "Cisco",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "asa",
  "semantic_action": "deny",
  "network_protocol": "udp",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": 46422,
  "dst_port": 53,
  "source_zone": "dmz-2",
  "destination_zone": "outside",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Cisco|ASA Firewall|asa",
  "normalized_message": "<<NUM>>Jul <NUM> <USER> <IPV6> <USER> Deny udp src dmz-<NUM>:<IP>/<NUM> dst outside:<IP>/<NUM> by <ORG> \"<USER>\" [0x0, 0x0]",
  "semantic_template": "format=asa action=deny protocol=udp",
  "semantic_field_count": 6,
  "is_auth_failure": 0,
  "is_network_denied": 1,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=53，field_token_count=59；每条编码上限96，原文长度156。这里的固定长度不代表全文所有字段均进入网络。

### EVT-0000134221 · Windows Active Directory

这是3893字符长日志，事件码4625与失败正文同时存在；原文字段名有脱敏扰动。完整原文与归一化片段不同，后者存在长度限制。

完整原始输入：

```json
{
  "event_id": "EVT-0000134221",
  "timestamp": 1721992263.8996067,
  "pipeline": "syslog",
  "src_ip": "192.168.147.151",
  "dst_ip": "",
  "src_port": "58382",
  "src_host": "USER-0010-0012.domain-0022.example.net",
  "dst_host": "",
  "username": "",
  "message_sanitized": "ORG-1657 ::: {\"@metaUSER-0010-54061\":{\"beat\":\"winlogbeat\",\"type\":\"_doc\",\"version\":\"8.2.2\"},\"@ORG-1526stamp\":\"USER-9546-07-26T11:10:08.448Z\",\"USER-0010-1577\":{\"ephemeral_id\":\"CRED-24CRED-25582381-3d10-4395-bc1c-5cdb52ecb35e\",\"id\":\"12e13a9a-252a-493f-9c35-fa4d61abd0bc\",\"name\":\"USER-0010-0206\",\"type\":\"winlogbeat\",\"version\":\"8.2.2\"},\"ecs\":{\"version\":\"8.0.0\"},\"USER-0010-56507\":{\"USER-0010-0121\":\"ORG-0106\",\"code\":\"4625\",\"created\":\"USER-9546-07-26T11:10:08.678Z\",\"kind\":\"USER-0010-56507\",\"outcome\":\"failure\",\"provider\":\"USER-8162-ORG-0407-Security-Auditing\"},\"USER-0010\":{\"name\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\"},\"log\":{\"level\":\"ORG-0706rmation\"},\"CRED-23501\":\"An account failed to log on.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tS-1-5-18\\n\\tAccount ORG-4643 ORG-CRED-25687\\USER-1430\\tORG-0106 ID:\\t\\t0x3E7\\n\\nORG-0106 Type:\\t\\t\\t3\\n\\nAccount For Which ORG-0106 Failed:\\n\\tSecurity ID:\\t\\tS-1-0-0\\n\\tAccount Name:\\t\\t\\n\\tAccount USER-24351:\\t\\t\\n\\USER-7624 ORG-0706rmation:\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\n\\tStatus:\\t\\t\\t0xC000006D\\n\\tSub Status:\\t\\t0xC0000064\\n\\USER-8912 ORG-0706rmation:\\n\\tCaller Process ID:\\t0x40c\\n\\tCaller Process Name:\\tC:\\\\ORG-0407\\\\USER-0010-1120\\\\ORG-3411\\USER-1430\\nORG-10379 ORG-0706rmation:\\n\\tWorkstation Name:\\tORG-0891-ORG-0023\\USER-1430\\USER-7625 ORG-10379 Address:\\t192.168.147.151\\n\\USER-7625 Port:\\t\\t58382\\n\\nDetailed ORG-0823 Process:\\t\\tSchannel\\n\\tAuthentication USER-0010-4214:\\tKerberos\\n\\tTransited USER-5935:\\t-\\n\\tUSER-0010-4214 Name (ORG-0504 only):\\t-\\n\\tKey Length:\\t\\t0\\n\\nUSER-9570 USER-0010-56507 is generated when a ORG-0106 USER-9484 fails. It is generated on the USER-3828 where ORG-1738 was atUSER-0010-1591ted.\\n\\nThe Subject fields indicate the account on the ORG-0462 USER-0086 which USER-9484ed the ORG-0106. USER-9570 is most USER-0010-54152ly a ORG-0090 such as the Server ORG-0090, or a ORG-0462 process such as ORG-0505 or USER-1847.\\n\\nThe ORG-0106 Type field indicates the kind of ORG-0106 that was USER-9484ed. The most USER-0010-54152 types are 2 (interCRED-24062) and 3 (ORG-10379).\\n\\nThe Process ORG-0706rmation fields indicate which account and process on the USER-0086 USER-9484ed ORG-2395 fields indicate where a ORG-4683 ORG-0106 USER-9484 originated. Workstation name is not always available and may be left blank in some cases.\\n\\nThe authentication ORG-0706rmation fields provide detailed ORG-0706rmation about USER-9570 specific ORG-0106 USER-9484.\\n\\t- Transited USER-5935 indicate which intermediate USER-5935 have participated in USER-9570 ORG-0106 ORG-0507 name indicates which sub-protocol was used among the ORG-0504 protocols.\\n\\t- Key length indicates the length of the generated session key. USER-9570 will be 0 if no session key was USER-9484ed.\",\"organization\":\"ORG-0003\",\"senderUSER-0010\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\",\"sensitivity\":\"normal\",\"winlog\":{\"api\":\"winUSER-0010-56507log\",\"channel\":\"Security\",\"USER-3828_name\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\",\"USER-0010-56507_USER-0010-54061\":{\"AuthenticationUSER-0010-4214Name\":\"Kerberos\",\"FailureReason\":\"%%2313\",\"IpAddress\":\"192.168.141.126\",\"IpPort\":\"58382\",\"KeyLength\":\"0\",\"LmUSER-0010-4214Name\":\"-\",\"ORG-0106ProcessName\":\"Schannel\",\"ORG-0106Type\":\"3\",\"ProcessId\":\"0x40c\",\"ProcessName\":\"C:\\\\ORG-0407\\\\USER-0010-1120\\\\USER-0010-1155\",\"Status\":\"0xc000006d\",\"SubStatus\":\"0xc0000064\",\"SubjectUSER-24351Name\":\"ORG-0015\",\"SubjectORG-0106Id\":\"0x3e7\",\"SubjectUserName\":\"USER-0109\",\"SubjectCRED-0339id\":\"S-1-5-18\",\"USER-0010-4999CRED-0339id\":\"S-1-0-0\",\"TransmittedUSER-5935\":\"-\",\"WorkstationName\":\"USER-0010-0206\"},\"USER-0010-56507_id\":\"4625\",\"keywords\":[\"Audit Failure\"],\"opcode\":\"ORG-0706\",\"process\":{\"pid\":1036,\"thread\":{\"id\":8604}},\"provider_ORG-0515\":\"{54849625-5478-4994-a5ba-3e3b0328c30d}\",\"provider_name\":\"USER-8162-ORG-0407-Security-Auditing\",\"record_id\":6067660825,\"task\":\"ORG-0106\"}}",
  "product_name": "Windows Active Directory",
  "vendor_name": "Microsoft",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "windows_json",
  "semantic_action": "fail",
  "network_protocol": "__MISSING__",
  "event_code": "4625",
  "event_name": "logon_failure",
  "src_port_from_message": -1,
  "dst_port": -1,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Microsoft|Windows Active Directory|windows_json",
  "normalized_message": "<ORG> <IPV6> {\"@meta<USER>\":{\"beat\":\"winlogbeat\",\"type\":\"_doc\",\"version\":\"<NUM>.<NUM>\"},\"@<ORG>\":\"USER-<TIMESTAMP>\",\"<USER>\":{\"ephemeral_id\":\"<CREDENTIAL>-<UUID>\",\"id\":\"<UUID>\",\"name\":\"<USER>\",\"type\":\"winlogbeat\",\"version\":\"<NUM>.<NUM>\"},\"ecs\":{\"version\":\"<NUM>.<NUM>\"},\"<USER>\":{\"<USER>\":\"<ORG>\",\"code\":\"<NUM>\",\"created\":\"USER-<TIMESTAMP>\",\"kind\":\"<USER>\",\"outcome\":\"failure\",\"provider\":\"<USER>\"},\"<USER>\":{\"name\":\"<USER>.<USER>.example<USER>\"},\"log\":{\"level\":\"<ORG>\"},\"<CREDENTIAL>\":\"An account failed to log on.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tS-<NUM>-<NUM>-<NUM>\\n\\tAccount <ORG> <ORG>\\<USER>\\t<ORG> ID:\\t\\t0x3E7\\n\\n<ORG> Type:\\t\\t\\t3\\n\\nAccount For Which <ORG> Failed:\\n\\tSecurity ID:\\t\\tS-<NUM>-<NUM>-<NUM>\\n\\tAccount Name:\\t\\t\\n\\tAccount <USER>:\\t\\t\\n\\<USER> <ORG>:\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\n\\tStatus:\\t\\t\\t0xC000006D\\n\\tSub Status:\\t\\t0xC0000064\\n\\<USER> <ORG>:\\n\\tCaller Process ID:\\t0x40c\\n\\tCaller Process Name:\\tC:\\\\<ORG>\\\\<USER>\\\\<ORG>\\<USER>\\n<ORG> <ORG>:\\n\\tWorkstation Name:\\t<ORG>\\<USER>\\<USER> <ORG> Address:\\t<IP>\\n\\<USER> Port:\\t\\t58382\\n\\nDetailed <ORG> Process:\\t\\tSchannel\\n\\tAuthentication <USER>:\\tKerberos\\n\\tTransited <USER>:\\t-\\n\\t<USER> Name (<ORG> only):\\t-\\n\\tKey Length:\\t\\t0\\n\\n<USER> <USER> is generated when a <ORG> <USER> fails. It is generated on the <USER> where <ORG> was at<USER>.\\n\\nThe Subject fields indicate the account on the <ORG> <USER> which <USER> the <ORG>. <USER> is most <USER> a <ORG> such as the Server <ORG>, or a <ORG> process such as <ORG> or <USER>.\\n\\nThe <ORG> Type field indicates the kind of <ORG> that was <USER>. The most <USER> types are <NUM> (inter<CREDENTIAL>) and <NUM> (<ORG>).\\n\\nThe Process <ORG> fields indicate which account and process on the <USER> <USER> <ORG> fields indicate where a <ORG> <ORG> <USER> originated. Workstation name is not always available and may be left blank in some cases.\\n\\nThe authentication <ORG> fields provide detailed <ORG> about <USER> specific <ORG> <USER>.\\n\\t- Transited <USER> indicate which intermediate <U",
  "semantic_template": "format=windows_json event_code=4625 event_name=logon_failure action=fail keys=beat,type,version,ephemeral_id,id,name,ecs,code,created,kind",
  "semantic_field_count": 2,
  "is_auth_failure": 1,
  "is_network_denied": 0,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=96，field_token_count=96；每条编码上限96，原文长度3893。这里的固定长度不代表全文所有字段均进入网络。

### EVT-0000441159 · Duo

认证字段的实际键名含脱敏串，不能把它写成未出现的event_type/factor。no_response、duo_push、denied和auth_failure作为值仍存在，动作解析缺失并不表示正文无信息。

完整原始输入：

```json
{
  "event_id": "EVT-0000441159",
  "timestamp": 1721992859.0292988,
  "pipeline": "syslog",
  "src_ip": "",
  "dst_ip": "",
  "src_port": "",
  "src_host": "HOST-0019",
  "dst_host": "",
  "username": "USER-1343",
  "message_sanitized": "ORG-1780 ::: streamName=ORG-2927 ::: tags=[no_response] ::: CRED-23501={\"ORG-1738_USER-0010-1127\":{\"epkey\":null,\"USER-0010name\":null,\"ip\":\"100.64.52.151\",\"location\":{\"city\":null,\"country\":null,\"USER-0010-54774\":null}},\"alias\":\"\",\"application\":{\"key\":\"DIBE3VETN055FUSIDPC8\",\"name\":\"AWS USER-CRED-30678 VPN\"},\"auth_USER-0010-1127\":{\"ip\":null,\"key\":\"DPSH5S416UGQQL9GWA6G\",\"location\":{\"city\":null,\"country\":null,\"USER-0010-54774\":null},\"name\":\"412-370-9347\"},\"USER-0010-1CRED-23741\":\"user-0819@exampleUSER-8710\",\"USER-0010-56507_type\":\"authentication\",\"fUSER-0010-15196\":\"duo_push\",\"isoORG-1526stamp\":\"USER-9546-07-26T11:00:57.963780+00:00\",\"ood_USER-0010-1219\":null,\"reason\":\"no_response\",\"result\":\"denied\",\"ORG-1526stamp\":17219CRED-CRED-2980737,\"trusted_USER-0010-CRED-29699_status\":\"unknown\",\"txid\":\"67ca3e81-5815-4ed1-92a1-3616fa888a39\",\"user\":{\"groups\":[],\"key\":\"DUJCLW3N9SAXF4WG6Y0A\",\"name\":\"USER-1413\"}} ::: userName=user-0819@exampleUSER-8710 ::: USER-0010-1129=USER-5841 ::: CRED-23501Type=auth_failure",
  "product_name": "Duo",
  "vendor_name": "Cisco",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "json",
  "semantic_action": "__MISSING__",
  "network_protocol": "__MISSING__",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": -1,
  "dst_port": -1,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Cisco|Duo|json",
  "normalized_message": "<ORG> <IPV6> streamName=<ORG> <IPV6> tags=[no_response] <IPV6> <CREDENTIAL>={\"<ORG>_<USER>\":{\"epkey\":null,\"<USER>\":null,\"ip\":\"<IP>\",\"location\":{\"city\":null,\"country\":null,\"<USER>\":null}},\"alias\":\"\",\"application\":{\"key\":\"DIBE3VETN055FUSIDPC8\",\"name\":\"AWS <USER> VPN\"},\"auth_<USER>\":{\"ip\":null,\"key\":\"DPSH5S416UGQQL9GWA6G\",\"location\":{\"city\":null,\"country\":null,\"<USER>\":null},\"name\":\"<NUM>-<NUM>-<NUM>\"},\"<USER>\":\"<USER>@example<USER>\",\"<USER>_type\":\"authentication\",\"f<USER>\":\"duo_push\",\"iso<ORG>\":\"USER-<TIMESTAMP>\",\"ood_<USER>\":null,\"reason\":\"no_response\",\"result\":\"denied\",\"<ORG>\":<NUM><CREDENTIAL>,\"trusted_<USER>_status\":\"unknown\",\"txid\":\"<UUID>\",\"user\":{\"groups\":[],\"key\":\"DUJCLW3N9SAXF4WG6Y0A\",\"name\":\"<USER>\"}} <IPV6> userName=<USER>@example<USER> <IPV6> <USER>=<USER> <IPV6> <CREDENTIAL>=auth_failure",
  "semantic_template": "format=json keys=epkey,ip,location,city,country,alias,application,key,name,auth_user-0010-1127",
  "semantic_field_count": 0,
  "is_auth_failure": 0,
  "is_network_denied": 0,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=96，field_token_count=96；每条编码上限96，原文长度1007。这里的固定长度不代表全文所有字段均进入网络。

### EVT-0000600002 · AWS VPC Security

末尾REJECT OK中的OK是记录状态。当前粗解析得到reject，但协议仍为__MISSING__；不能因为第七列有6就声称此解析器实际输出tcp。

完整原始输入：

```json
{
  "event_id": "EVT-0000600002",
  "timestamp": 1721993140.7599907,
  "pipeline": "syslog",
  "src_ip": "100.64.0.237",
  "dst_ip": "10.216.192.18",
  "src_port": "48165",
  "src_host": "HOST-0031",
  "dst_host": "",
  "username": "",
  "message_sanitized": "2 100000013063 ORG-1504 100.64.0.237 10.182.224.117 48165 39878 6 1 40 1721992874 1721CRED-2CRED-3023300 REJECT OK",
  "product_name": "AWS VPC Security",
  "vendor_name": "Amazon Web Services",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "vpc_flow",
  "semantic_action": "reject",
  "network_protocol": "__MISSING__",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": 48165,
  "dst_port": 39878,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Amazon Web Services|AWS VPC Security|vpc_flow",
  "normalized_message": "<NUM> <NUM> <ORG> <IP> <IP> <NUM> <NUM> <NUM> <NUM> <NUM> <NUM> <NUM><CREDENTIAL> REJECT OK",
  "semantic_template": "format=vpc_flow action=reject",
  "semantic_field_count": 3,
  "is_auth_failure": 0,
  "is_network_denied": 1,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=37，field_token_count=41；每条编码上限96，原文长度114。这里的固定长度不代表全文所有字段均进入网络。

## 5. 时间外推、OOF与下一阶段


已经由实验支持的结论：Drain/确定性解析能显著增强结构基线；把过细 template/schema ID
当特征会形成查表风险；固定哈希正文能学习威胁含义，但三分类单头会混淆“威胁检测”和
“业务子类型”；分层模型实验取得本项目单网络最大的整体改进；整塔多视图替换不稳定；冻结锚点加局部可信
域能在不新增错误的情况下修复 content 与融合冲突。

尚未证明：v5.2 在未来时间或隐藏格式上仍保持 21/0 的修正精度；11 个全分支低分 FN 能否
仅靠新内容表示修复；24 个 subtype 错是否可在不动 threat 边界时解决。

下一阶段固定 v5.2，先只在原训练数据中选择一个全局时间截点 t：timestamp<t 用作训练，
timestamp>=t 用作开发验证。每类分别取最新百分比会产生不同截点，可能让某类训练行晚于
另一类验证行，因而不能称为严格的未来外推；它只能作为额外的按类晚期分布测试。若统一截点
后某类太少，应披露支持数，并增加滚动时间窗口，不能悄悄把未来行移回训练。

重复日志家族横跨截点时，记录重叠并增设去重/家族隔离对照；对需要严格隔离的主体，删除
跨界重复或设置间隔窗口，避免跨界信息。训练窗口内拟合所有预处理器、频次与参数。
OOF 表示每条训练行的锚点预测来自未用过该行及其隔离组训练的折模型，再用这些预测训练或
校准可信度。当前尚无全量 OOF 或统一时间截点实测结果，不能登记为已完成改进。

合并官方训练和验证标签只适用于所有方案冻结后的最终拟合；此后原验证集已经参与训练，
不能再拿它证明泛化。后续评价需另一个从未参与选择的测试集。公开验证已反复用于 epoch、
规则、gap 与 seed 选择，现有成绩应解释为开发验证成绩。


## 6. 本地测试与云端恢复


v4.1曾用真实小样本1,500训练/600验证跑三组各1个CPU epoch，检查模型保存/重载、
Parquet连接、错误摘要和覆盖审计；40项测试通过。小样本分数没有作为正式效果。
v5.0烟雾用45,620训练/11,517验证，两个epoch0均Score0.9994781085、10错；
54个实际冲突全部benign，训练后回退，44项测试通过。
v5.1同范围exp01回退epoch0；exp02 epoch1 Score0.9995468423，威胁FN2→1、FP保持1，
三分类仍10错，因为malicious→benign变成malicious→suspicious。
由此补充了fixed_anchor_threat_errors与new_threat_errors，不能只看三分类fixed。

云端曾出现脚本不存在（未切换含脚本分支）、原始数据位置不符（实际在/root/work）、
checkpoint丢失、重复启动同输出目录、只见nohup ignoring input却误以为训练完成、
metrics尚未生成就读取等工程问题。日志出现Terminated只证明进程终止，不能未经系统日志
确认就认定OOM。处理顺序是定位数据与分支、检查进程/完整产物、读取训练日志，再恢复唯一任务。
避免同时两个训练进程写同一输出目录。用tail查看时Ctrl+C只结束查看，不应反复启动训练。

产物必须外部备份：权重、预处理器、模板状态（解析版本）、metrics、预测、manifest、
环境和提交号一起保存，生成SHA-256后复制到持久存储，下载后再校验。Git代码提交不包含云端权重。
文件哈希确认字节身份；特征语义比较还需按event_id对齐后比较列值与行顺序。

## 7. 代码与产物治理


main 仅提供环境、数据约定、评分公式、开发日志和版本导航，不存放训练源码。
每个正式版本有独立分支和独立 README：包含输入、模型、训练命令、该版本结果、缺陷和后续
方法。各分支只保留其训练、推理、审计及必要共享依赖，不堆放其他模型家族的独立入口。
版本只用 v主版本.次版本；对照用 expNN，随机重复用 seed，权重轮次用 ep。

当前 v5.2 默认入口只跑 exp01。每次训练保存 `model.pt`、`metrics.json`、
`valid_predictions.parquet`、预处理器或嵌入式预处理元数据、特征 manifest、环境信息、
命令参数、源码提交号及 SHA-256。模型权重包含数值状态，指标只是评价记录，后者不能恢复前者。
仅模型、指标、预测三个文件存在是脚本的跳过条件，不足以证明参数和数据身份一致；人工复用时
仍应核对 manifest 和哈希。特征改列名后必须重建对应特征并重训，不能只改文件名冒充兼容。

## 8. 全部字段的含义与派生规则


以下列表从训练代码实际声明生成；每行均说明字段含义。输出Parquet中的审计列不自动成为模型输入。
原始event_id、label_binary、完整时间、原始IP/主机/账号字符串不是神经网络身份特征。

#### v1.0及v2.x结构基础输入

##### 类别字段（8个）

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

##### 数值字段（23个）

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

#### v1.1完整输入

##### 类别字段（22个）

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

##### 数值字段（41个）

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

#### v1.2完整输入

##### 类别字段（36个）

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

##### 数值字段（58个）

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

#### v4.x/v5.x元数据塔完整输入

##### 类别字段（9个）

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

##### 数值字段（23个）

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

#### v4.x/v5.x语义塔完整输入

##### 类别字段（4个）

| 字段 | 含义 |
|---|---|
| `content_family` | 内容解析家族。 |
| `content_action` | 内容粗动作。 |
| `content_protocol` | 内容协议。 |
| `content_event_code` | 内容事件码。 |

##### 数值字段（4个）

| 字段 | 含义 |
|---|---|
| `content_has_threat` | 内容安全词组的威胁信号；不是标签。 |
| `content_has_authentication` | 认证关键词信号。 |
| `content_has_potentially_harmful` | potentially harmful相关词组信号。 |
| `raw_token_count` | raw序列非padding数量，上限96。 |

#### 内容序列与残差标量

v3.0-exp01只输入`raw_token_ids`。exp02输入上述v1.0全部8类别/23数值加`raw_token_ids`；exp03替换为`field_token_ids`。
v4.0用元数据塔9类别/23数值、语义塔4类别/4数值及`raw_token_ids`；训练频次键为`product_name`、`content_family`、`content_action`。
v4.1-exp01/exp03改用`multiview_token_ids`，exp02仍用raw。四视图顺序head、middle、tail、key_value，每视图64。
v5.x冻结v4.0三塔与分类头，可信度输入为metadata128、semantic64、content128、metadata_margin、content_margin、novelty_gate、log1p(combo_count)，共324维；附加四视图实验再加128维，共452维。
anchor_margin仅用于候选与gap计算，不作为trust网络输入。全部独立输入如上，没有省略的用户ID或隐藏标签输入。
