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
