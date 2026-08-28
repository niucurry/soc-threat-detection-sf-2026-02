# 云平台上传与V1训练说明

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

## 4. 启动训练

在 `soc_cloud_v1` 目录执行：

```bash
bash scripts/run_cloud_v1.sh
```

脚本会依次：

1. 安装不包含PyTorch的辅助依赖，避免破坏镜像自带的PyTorch-NPU组合；
2. 检查NPU是否可用；
3. 将长日志转换成紧凑的结构特征；
4. 在NPU上训练V1模型；
5. 保存最佳模型、完整验证指标和每条验证日志的预测概率。

## 5. 需要发回的结果

训练结束后，请下载或把以下三个小文件提供给Codex：

```text
artifacts/v1_npu_tabular/environment.json
artifacts/v1_npu_tabular/train_console.log
artifacts/v1_npu_tabular/metrics.json
```

模型文件 `model.pt` 和两百万条验证预测可以先留在云端，不必立即下载。

## 常见处理

- 如果提示NPU不可用：检查实例是否实际挂载NPU，并确认没有更换镜像中的PyTorch或torch_npu。
- 如果提示内存不足：将脚本中的 `--batch-size 8192` 改成 `4096`。
- 如果依赖无法联网安装：保留完整报错日志，不要自行升级PyTorch，把报错发回再调整。
- 如果训练中断：特征文件会保留，再次执行脚本会跳过预处理并重新训练。

