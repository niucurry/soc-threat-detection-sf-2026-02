# SOC日志三分类项目

目标：将每条日志分类为 `benign`（正常）、`malicious`（恶意）或
`suspicious`（可疑）。所有实验都遵守两个原则：不使用 `event_id` 猜标签，
并始终在官方提供的私有验证标签上报告三类分别的效果。

## 推荐赛方镜像

使用：`pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64`。

它适合训练并提供 Jupyter/VS Code。第一版云端主模型使用原生 PyTorch，
自动优先选择 `npu:0`；同一套代码在没有 NPU 时也能退回 CUDA 或 CPU 做小规模检查。

## 当前版本：V3 语义增强混合模型

V3 延续 V2 的分层检测器，并针对未见日志格式做了泛化审计：

1. PyTorch 结构模型负责全部日志的三分类基础判断；
2. 对 `pipeline=syslog` 且产品名为空的混合簇，使用日志正文
   TF-IDF + SGD 二分类器重新判断正常/恶意；
3. 对 `REJECT OK`、Windows 4625、明确的防火墙拒绝/丢弃、URL 阻断等
   高置信安全语义使用规则兜底；
4. 提供保守版和使用少量可疑事件规则的调优版，并严格校验验证预测的
   `event_id` 完整性。

训练集恶意占比约 5.43%，官方验证集约 0.70%。V1 结构特征无法区分
产品名为空的正常和恶意日志，导致 9,793 条正常日志被误报为恶意；V2 的
正文专用模型用于解决这个问题。

正式评分按下式计算：

```text
Final Score = 0.40 × Threat-Binary-F1
            + 0.25 × Threat-Binary-Recall
            + 0.15 × Threat Recall
            + 0.10 × Macro-F1
            + 0.05 × Soft Label Score
            + 0.05 × Balanced Accuracy
```

其中 Threat-Binary 将 `suspicious` 和 `malicious` 合并为威胁类；Threat
Recall 是两种威胁召回率的平均值。当前代码根据评分说明把 Soft Label Score
实现为逐行平均：预测完全正确计 1，`suspicious` 与 `malicious` 相互错判计
0.5，正常与威胁之间错判计 0。正文专模阈值和结构模型最佳轮次均优先按照
Final Score 选择，不再只优化 Macro-F1。

在提供的 2,014,052 条官方外部验证数据上，本地复现实验如下：

| 版本 | 正式综合分 | Macro-F1 | 错误行数 | 说明 |
|---|---:|---:|---:|---|
| V1 无类别权重 | 0.957210 | 0.912286 | 9,857 | 原结构模型 |
| V3 保守版 | 0.999890 | 0.999958 | 10 | 恶意语义增强，不使用可疑事件覆盖规则 |
| V3 调优版 | 1.000000 | 1.000000 | 0 | 再增加 DLP 和 Duo 高精度可疑规则 |

这些是已提供验证标签上的结果，不代表隐藏数据一定满分。调优版更贴合当前
验证分布，保守版对新数据源的假设更少，后续仍需通过隔离验证比较泛化能力。

V3 审计了 407 万条训练/验证日志。新增的 `TRAFFIC,drop` 和
`THREAT,url + block-url` 语义分别覆盖 5,826 和 200 条验证恶意日志，正常命中
均为 0。高置信恶意规则总覆盖由 8,024 提升到 14,050，正文决策阈值由约
0.057 提升到 0.175。剩余两条 Windows 4672 事件继续交给正文模型判断，因为
该事件码在真实环境中不必然代表攻击。

## V3 云端复核

三份数据仍按下列文件名放置：

```text
data/raw/train.parquet
data/raw/valid_input.parquet
data/raw/valid_answer_private.parquet
```

V3 只更新正文规则层和阈值，直接复用已经完成的 V2 结构模型验证预测，避免
重复执行 NPU 训练。在推荐镜像中使用后台方式运行：

```bash
mkdir -p artifacts/v3_semantic_rules
nohup bash scripts/run_cloud_v3.sh data/raw \
  > artifacts/v3_semantic_rules/nohup.log 2>&1 &
echo $! > artifacts/v3_semantic_rules/train.pid
cat artifacts/v3_semantic_rules/train.pid
```

如果数据位于其他目录：

```bash
mkdir -p artifacts/v3_semantic_rules
nohup bash scripts/run_cloud_v3.sh /root/work \
  > artifacts/v3_semantic_rules/nohup.log 2>&1 &
echo $! > artifacts/v3_semantic_rules/train.pid
cat artifacts/v3_semantic_rules/train.pid
```

脚本默认复用 `artifacts/v2_hybrid/base/valid_predictions.parquet`。如果 V2 基础
预测在其他位置，可在命令前设置 `V3_BASE_PREDICTIONS=/实际路径/valid_predictions.parquet`。

查看总进度日志：

```bash
tail -f artifacts/v3_semantic_rules/nohup.log
```

按 `Ctrl+C` 只退出日志查看，不会终止后台训练。检查进程和 NPU：

```bash
PID=$(cat artifacts/v3_semantic_rules/train.pid)
ps -fp "$PID"
npu-smi info
```

正文模型开始训练后，还可以单独查看输出：

```bash
tail -f artifacts/v3_semantic_rules/text/train_console.log
```

主要结果位于：

```text
artifacts/v3_semantic_rules/text/model.joblib
artifacts/v3_semantic_rules/validation_conservative/metrics.json
artifacts/v3_semantic_rules/validation_tuned/metrics.json
```

## V1 结构基线

V1 云端主线是“类别嵌入 + 数值网络”的 PyTorch 结构化模型：

- 不使用 `event_id`、完整时间、原始 IP、原始主机名或原始用户名；
- 使用产品、采集管道、端口范围、字段缺失情况、消息长度和少量关键词；
- 使用可调节的类别平衡权重，避免模型只预测占比最大的正常类别；
- 主要评价指标为 Macro-F1，同时报告每一类召回率和混淆矩阵。

当前上传包只包含PyTorch-NPU主线代码，避免在ARM镜像中安装不必要的CPU模型依赖。

## V1 上传目录

把三份原始文件放到：

```text
data/raw/train.parquet
data/raw/valid_input.parquet
data/raw/valid_answer_private.parquet
```

随后在项目根目录运行：

```bash
bash scripts/run_cloud_v1.sh
```

如果三份原始文件已经直接放在云平台的 `/root/work`，不需要复制文件，运行：

```bash
bash scripts/run_cloud_v1.sh /root/work
```

正式训练前可在相同的20万训练/验证样本上比较四种类别补偿强度：

```bash
bash scripts/run_weight_sweep_v1.sh
```

汇总结果保存在 `artifacts/v1_weight_sweep/comparison.json`，用于选择能兼顾
攻击召回与误报数量的正式参数。

训练日志会实时显示，并保存到
`artifacts/v1_npu_tabular/train_console.log`。最终模型、完整指标和验证集预测
也会保存在同一目录。

详细上传和故障处理步骤见 `UPLOAD_INSTRUCTIONS.md`。
