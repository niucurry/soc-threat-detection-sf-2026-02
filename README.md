# SOC日志三分类项目

目标：将每条日志分类为 `benign`（正常）、`malicious`（恶意）或
`suspicious`（可疑）。所有实验都遵守两个原则：不使用 `event_id` 猜标签，
并始终在官方提供的私有验证标签上报告三类分别的效果。

## 推荐赛方镜像

使用：`pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64`。

它适合训练并提供 Jupyter/VS Code。第一版云端主模型使用原生 PyTorch，
自动优先选择 `npu:0`；同一套代码在没有 NPU 时也能退回 CUDA 或 CPU 做小规模检查。

## 当前版本：V2 混合模型

V2 针对官方验证集暴露出的分布漂移增加了一个分层检测器：

1. PyTorch 结构模型负责全部日志的三分类基础判断；
2. 对 `pipeline=syslog` 且产品名为空的混合簇，使用日志正文
   TF-IDF + SGD 二分类器重新判断正常/恶意；
3. 对 `REJECT OK`、Windows 4625、明确的防火墙阻断等高置信安全语义
   使用规则兜底；
4. 提供保守版和使用少量可疑事件规则的调优版，并严格校验提交文件的
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
| V2 保守版 | 0.999890 | 0.999958 | 10 | 不使用新增可疑事件覆盖规则 |
| V2 调优版 | 1.000000 | 1.000000 | 0 | 增加 DLP 和 Duo 高精度可疑规则 |

这些是已提供验证标签上的结果，不代表隐藏测试集一定满分。调优版更贴合当前
验证分布，保守版对新数据源的假设更少，正式提交时建议两版都保留并结合榜单
反馈选择。

## V2 云端训练

三份数据仍按下列文件名放置：

```text
data/raw/train.parquet
data/raw/valid_input.parquet
data/raw/valid_answer_private.parquet
```

在推荐 NPU 镜像中运行：

```bash
bash scripts/run_cloud_v2.sh
```

如果数据位于其他目录：

```bash
bash scripts/run_cloud_v2.sh /root/work
```

主要结果位于：

```text
artifacts/v2_hybrid/base/model.pt
artifacts/v2_hybrid/text/model.joblib
artifacts/v2_hybrid/validation_conservative/metrics.json
artifacts/v2_hybrid/validation_tuned/metrics.json
```

## 测试集预测与 res.csv

测试集发布后运行：

```bash
bash scripts/run_inference_v2.sh /测试集绝对路径/test.parquet artifacts/v2_submission/res.csv
```

默认生成调优版结果。如果要生成保守版：

```bash
V2_RULE_MODE=conservative bash scripts/run_inference_v2.sh \
  /测试集绝对路径/test.parquet artifacts/v2_submission/res_conservative.csv
```

提交脚本会检查测试集和预测文件的行数、重复 ID、缺失 ID、多余 ID 和标签
枚举值，最终 CSV 只包含 `event_id,pred_label` 两列。

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

汇总结果保存在 `artifacts/v1_weight_sweep/comparison.csv` 和
`comparison.json`，用于选择能兼顾攻击召回与误报数量的正式参数。

训练日志会实时显示，并保存到
`artifacts/v1_npu_tabular/train_console.log`。最终模型、完整指标和验证集预测
也会保存在同一目录。

详细上传和故障处理步骤见 `UPLOAD_INSTRUCTIONS.md`。

## 大型CSV/CVS文件打不开时

不要使用Excel直接打开数百万行日志。先运行轻量检查命令，它只读取文件开头，
不会把整个文件载入内存：

```bash
python src/inspect_data_file.py /实际路径/数据文件.cvs
```

将命令输出发回后，再根据真实格式决定是否重命名、解压、拆分或修改预处理代码。

如果文件头是 `system,prompt,response`，它属于指令微调数据。训练文件按标签相关顺序排列，
因此需要流式扫描完整文件才能得到可靠的标签分布：

```bash
python src/analyze_sft_csv.py /实际路径/train_system_prompt_response.csv \
  --max-rows 0 \
  --output artifacts/sft_csv_sample_analysis.json
```

该脚本会正确处理prompt中的引号和换行，并统计响应标签、异常行和prompt长度。

## 只有system/prompt/response训练CSV时

直接运行SFT版V1脚本，参数是5GB CSV的绝对路径：

```bash
bash scripts/run_sft_cloud_v1.sh \
  "/root/work/基于SOC日志网络安全威胁检测算法设计与实现/train_system_prompt_response.csv"
```

脚本会流式解析prompt、恢复结构字段，并使用prompt哈希划分90%训练和10%内部验证。
相同prompt始终进入同一部分，避免重复日志同时出现在训练集和验证集中。

如已将私有官方验证仓库克隆到云平台，可在启动前设置：

```bash
export OFFICIAL_VALID_PATH=/私有验证仓库路径/data/v1_valid.parquet
```

训练结束后脚本会自动执行一次外部验证，并把结果写入
`artifacts/v1_sft_npu_tabular/official_validation/official_metrics.json`。
