# V5.1 第一阶段：结构化日志解析神经网络

## 1. 这一阶段究竟测试什么

V5.1 只改造第一个 PyTorch/NPU 神经网络基础模型，不启用 V2 正文
TF-IDF 专模，不启用 V3 语义规则，也不使用人工覆盖。因此最终
`metrics.json` 反映的是新数据处理方法对基础神经网络的单独影响。

整条处理链是：

```text
原始一行日志
  -> 按日志格式路由
  -> JSON / XML / CEF / VPC 结构解析，或自由文本 Drain
  -> 结构 schema + 安全语义 + 模板统计
  -> 与 V1 原有结构特征按 event_id 一对一合并
  -> 类别嵌入 + 数值标准化
  -> 256 -> 128 -> 64 的 PyTorch 多层感知机
  -> benign / malicious / suspicious
```

## 2. 各种日志如何处理

### JSON 和 JSON 外层封装

不再把整段 JSON 交给 Drain。解析器会在日志中找到完整或嵌入的 JSON，
递归展开对象和数组；对 `::: key=value` 形式的外层封装也单独抽取。
例如数据中的真实错误样本 `EVT-0002001511` 能得到：

```text
message_format = json
payload_parse_status = success
event_category_v5 = malware
event_action_v5 = block
event_reason_v5 = malware
event_severity_number = 2
threat_category_v5 = malware
```

Duo 样本 `EVT-0002002674` 能得到：

```text
event_category_v5 = authentication
event_type_v5 = authentication
authentication_factor = duo_push
application_name_v5 = lastpass
service_name_v5 = duo
```

### Windows XML

首先使用严格 XML 树解析。实际数据的脱敏器有时会替换 XML 标签名或
属性名，使它不再是合法 XML；此时使用有界的叶节和属性容错扫描。
这个退化路径不猜测标签含义，只恢复仍然存在的结构值。

真实的 `EVT-0002000017` 保留：

```text
event_code = 4672
event_name = special_privileges
structured_parser = json_recursive
payload_parse_status = success
```

4672 表示特殊权限登录，但不等于必然恶意，所以 V5.1 只把它作为模型
特征，不写死标签。

### CEF

先拆分 CEF 头部的 vendor/product/signature/name/severity，再按 CEF 扩展字段
拆分 `act`、`src`、`dst`、`spt`、`dpt`、`msg` 等。Drain 只看事件名和
`msg` 的剩余文本，不再被整串变动的键值对干扰。

真实 CEF 样本 `EVT-0000017315` 能得到：

```text
event_code = 342
event_action_v5 = deny
dst_port_number = 80
http_method = GET
rule_name_v5 = geo_ip_block
message_template 只包含 GEO_IP_BLOCK 和 GeoIP Match 剩余文本
```

### VPC Flow、ASA 和其他文本

VPC Flow 按固定位置解析源/目标地址、端口、协议号、action 和 status。
ASA、Linux Syslog、普通 Syslog、键值文本和自由文本继续使用 Drain，但
按 `pipeline|vendor|product|format` 分组，避免不同产品的词位被混在一起。

## 3. 新增特征的明确含义

- `schema_id`：结构化载荷的字段路径集合的稳定哈希，不包含字段值。
- `semantic_template_id`：格式、schema、事件码、类别、动作、结果等的语义组合哈希。
- `*_seen_train`：验证日志的模板/schema/语义模板是否在训练集出现。
- `*_frequency_log1p`：该结构在训练集的对数频率。
- `payload_parse_status`：`success`、`partial`、`failed`、`not_applicable` 或 `blank`。
- `event_category_v5` 等字段：只保留可确定的安全语义；单独的
  `USER-*`、`ORG-*`、`HOST-*`、`CRED-*` 脱敏令牌不会被当作语义类别。

## 4. 如何防止验证泄漏

Drain 聚类、schema 频率、语义模板频率、神经网络类别词表和数值
标准化参数都只在 `train.parquet` 上拟合。`valid_input.parquet` 只使用
已冻结的解析模型做匹配；`valid_answer_private.parquet` 只在计算指标时使用。

`event_id`、原始 IP、原始用户名和完整时间都不会输入神经网络。

## 5. 云平台完整运行

先切换到本分支：

```bash
git fetch origin
git switch feat/v5-structured-parsing
git pull --ff-only
```

三个 parquet 都在 `/root/work` 时：

```bash
mkdir -p artifacts/v5_structured_neural

nohup bash scripts/run_cloud_v5_structured_neural.sh /root/work \
  > artifacts/v5_structured_neural/nohup.log 2>&1 &

echo $! > artifacts/v5_structured_neural/train.pid
tail -f artifacts/v5_structured_neural/nohup.log
```

`tail -f` 中按 `Ctrl+C` 只会退出日志观看，不会终止后台任务。查看进程和
NPU：

```bash
PID=$(cat artifacts/v5_structured_neural/train.pid)
ps -fp "$PID"
npu-smi info
```

如果之前已经生成过 V5.1 中间特征，需要强制重建：

```bash
V5_FORCE_PREPARE=1 bash scripts/run_cloud_v5_structured_neural.sh /root/work
```

## 6. 训练完成后请回传什么

请依次执行：

```bash
cat artifacts/v5_structured_neural/base/metrics.json
cat artifacts/v5_structured_neural/analysis/error_summary.json
cat data/processed/v5/v5_manifest.json
wc -l artifacts/v5_structured_neural/analysis/error_rows.csv
head -30 artifacts/v5_structured_neural/analysis/error_rows.csv
```

如果 V4 预测文件在默认位置：

```text
artifacts/v4_drain_neural/base/valid_predictions.parquet
```

`error_summary.json` 中会额外出现 `v4_v5_comparison`，其中最重要的是：

- `v4_errors`：V4 基础神经网络错误数，预期为 76。
- `v5_errors`：V5.1 基础神经网络错误数。
- `v4_wrong_v5_correct`：V5.1 修复的 V4 错误。
- `v4_correct_v5_wrong`：V5.1 新引入的错误。
- `both_wrong`：两个基础模型都分错的行。

只有同时比较这些数字和错误行的语义，才能判断改善是真正的泛化
改善，还是只对当前验证集的偶然拟合。

## 7. 本地验证状态

- 24 个单元测试全部通过。
- 已用真实 parquet 抽样完成 5,000 条训练、1,000 条验证的端到端闭环。
- 闭环包括 V1 特征、V5.1 解析、模型保存/加载、CPU 训练、预测和错误
  审计文件生成。
- 抽样数据只用于验证代码链路，不能代替云端全量对比。
