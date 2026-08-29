# 云平台 V3 增量复核说明

## 1. 选择镜像

请选择：

`pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64`

创建实例时确认分配了至少一张NPU。建议内存不少于16GB、可用磁盘不少于20GB。

## 2. 上传并解压代码

将 `soc_cloud_v1.zip` 上传到实例，然后执行：

```bash
unzip soc_cloud_v1.zip -d soc_cloud_v1
cd soc_cloud_v1
mkdir -p data/raw
```

## 3. 放置三份数据

将数据上传到以下位置，文件名不要修改：

```text
soc_cloud_v1/data/raw/train.parquet
soc_cloud_v1/data/raw/valid_input.parquet
soc_cloud_v1/data/raw/valid_answer_private.parquet
```

可以用下面的命令核对：

```bash
ls -lh data/raw
```

## 4. 启动 V3 增量复核

V3 复用已经完成的 V2 结构模型预测，只重新训练正文专模并执行语义规则层。
在项目目录后台执行：

```bash
mkdir -p artifacts/v3_semantic_rules
nohup bash scripts/run_cloud_v3.sh /原始数据目录 \
  > artifacts/v3_semantic_rules/nohup.log 2>&1 &
echo $! > artifacts/v3_semantic_rules/train.pid
cat artifacts/v3_semantic_rules/train.pid
```

查看总日志：

```bash
tail -f artifacts/v3_semantic_rules/nohup.log
```

按 `Ctrl+C` 只停止日志跟随，不会停止训练。查看后台进程和 NPU 状态：

```bash
PID=$(cat artifacts/v3_semantic_rules/train.pid)
ps -fp "$PID"
npu-smi info
```

正文模型开始后可查看训练输出：

```bash
tail -f artifacts/v3_semantic_rules/text/train_console.log
```

脚本会依次：

1. 安装不包含PyTorch的辅助依赖，避免破坏镜像自带的PyTorch-NPU组合；
2. 检查运行环境；
3. 核验 V2 基础预测是否存在；
4. 在CPU上复训 V3 日志正文专用模型；
5. 应用扩展后的高精度拒绝、丢弃与 URL 阻断语义；
6. 分别保存保守版和调优版的完整验证指标与预测结果。

## 5. 需要发回的结果

训练结束后，请下载或把以下三个小文件提供给Codex：

```text
artifacts/v3_semantic_rules/environment.json
artifacts/v3_semantic_rules/text/train_console.log
artifacts/v3_semantic_rules/validation_conservative/metrics.json
artifacts/v3_semantic_rules/validation_tuned/metrics.json
```

正文模型 `artifacts/v3_semantic_rules/text/model.joblib` 和两百万条验证预测可以先留在云端，
不必立即下载。

## 常见处理

- 如果提示基础预测不存在：先确认 V2 产物目录；若位置不同，设置
  `V3_BASE_PREDICTIONS=/实际路径/valid_predictions.parquet` 后再启动。
- 如果提示NPU不可用：V3 正文训练本身使用 CPU，但仍应确认没有意外更换原镜像依赖。
- 如果依赖无法联网安装：保留完整报错日志，不要自行升级PyTorch，把报错发回再调整。
- 如果训练中断：V2 结构预测不会受影响，重新运行 V3 脚本即可。
