# SOC日志三分类项目

目标：将每条日志分类为 `benign`（正常）、`malicious`（恶意）或
`suspicious`（可疑）。所有实验都遵守两个原则：不使用 `event_id` 猜标签，
并始终在官方提供的私有验证标签上报告三类分别的效果。

## 推荐赛方镜像

使用：`pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64`。

它适合训练并提供 Jupyter/VS Code。第一版云端主模型使用原生 PyTorch，
自动优先选择 `npu:0`；同一套代码在没有 NPU 时也能退回 CUDA 或 CPU 做小规模检查。

## 当前版本

V1 云端主线是“类别嵌入 + 数值网络”的 PyTorch 结构化模型：

- 不使用 `event_id`、完整时间、原始 IP、原始主机名或原始用户名；
- 使用产品、采集管道、端口范围、字段缺失情况、消息长度和少量关键词；
- 使用可调节的类别平衡权重，避免模型只预测占比最大的正常类别；
- 主要评价指标为 Macro-F1，同时报告每一类召回率和混淆矩阵。

当前上传包只包含PyTorch-NPU主线代码，避免在ARM镜像中安装不必要的CPU模型依赖。

## 上传目录

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

训练日志会实时显示，并保存到
`artifacts/v1_npu_tabular/train_console.log`。最终模型、完整指标和验证集预测
也会保存在同一目录。

详细上传和故障处理步骤见 `UPLOAD_INSTRUCTIONS.md`。
