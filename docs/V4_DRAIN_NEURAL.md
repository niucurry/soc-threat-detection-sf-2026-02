# V4 混合日志解析与 Drain 神经网络实验

## 实验目的

V4 只替换第一层基础模型的输入特征，基础分类器仍是与 V1 相同的
PyTorch 类别嵌入 + 数值 MLP。V2 的 TF-IDF 专模和 V3 的规则均不参与本轮，
避免无法判断提升来自新日志模板，还是来自第二模型和人工规则。

## 特征处理

模板模型只在训练集上拟合：

```text
train.parquet
  -> 格式识别
  -> 结构化解析或分组 Drain
  -> 冻结 template_model

valid_input.parquet
  -> 使用同一格式识别
  -> 只匹配冻结模板
  -> 未匹配模板标为 drain_unmatched
```

确定性解析负责 Windows XML/JSON、CEF、ASA、VPC Flow 和 HTTP 等明确格式，
提取 `semantic_action`、`network_protocol`、`event_code`、`event_name`、
`src_port_from_message`、`dst_port_number`、HTTP 方法/状态和网络区域。

Drain 负责 ASA、Linux、CEF、键值对和普通 Syslog/自由文本。原有匿名令牌、IP、
UUID、邮箱、URL、MAC、长十六进制串和数字会先被类型化掩码。每个模板用
`SHA1(parser_group + template)` 生成稳定 `template_id`，不会把 Drain 的任意
cluster 编号当作连续数值输入神经网络。

空正文仍保留 `message_missing`，但当前训练集空正文全为正常事件，验证分布又有
明显变化，因此评估时必须重点检查模型是否形成“空正文就是正常”的捷径。

## 云平台正式训练

数据目录必须包含：

```text
train.parquet
valid_input.parquet
valid_answer_private.parquet
```

推荐后台执行：

```bash
cd /path/to/soc-threat-detection-sf-2026-02
mkdir -p artifacts/v4_drain_neural
nohup bash scripts/run_cloud_v4_drain_neural.sh /path/to/data \
  > artifacts/v4_drain_neural/nohup.log 2>&1 &
echo $! > artifacts/v4_drain_neural/train.pid
```

查看状态：

```bash
tail -f artifacts/v4_drain_neural/nohup.log
PID=$(cat artifacts/v4_drain_neural/train.pid)
ps -fp "$PID"
npu-smi info
```

首次运行会依次生成 V1 基础特征、拟合训练集模板、生成 V4 特征并训练 NPU 模型。
再次运行会复用现有特征。原始数据或解析代码发生变化后，强制重建：

```bash
V4_FORCE_PREPARE=1 bash scripts/run_cloud_v4_drain_neural.sh /path/to/data
```

可覆盖训练参数：

```bash
V4_EPOCHS=25 \
V4_BATCH_SIZE=8192 \
V4_LEARNING_RATE=0.002 \
V4_CLASS_WEIGHT_POWER=0.0 \
bash scripts/run_cloud_v4_drain_neural.sh /path/to/data
```

第一轮建议保持 `V4_CLASS_WEIGHT_POWER=0.0`，与 README 中的 V1 无类别权重结果
直接比较。只有确认模板特征本身的收益后，再单独扫描类别权重。

## 小规模烟雾测试

在 CPU/CUDA 环境可先验证解析和训练链路。若已经有完整 V1 特征：

```bash
python src/prepare_v4_features.py \
  --data-dir /path/to/data \
  --base-feature-dir data/processed \
  --output-dir data/processed/v4_smoke \
  --model-dir artifacts/v4_smoke/template_model \
  --max-train-rows 20000 \
  --max-valid-rows 20000 \
  --force

python src/train_npu_tabular.py \
  --feature-set v4 \
  --train data/processed/v4_smoke/v4_train.parquet \
  --valid data/processed/v4_smoke/v4_valid.parquet \
  --device cpu \
  --epochs 2 \
  --batch-size 1024 \
  --num-workers 0 \
  --output-dir artifacts/v4_smoke/base
```

烟雾测试只验证代码是否可运行，分层抽样行数很小时不能用于判断模型效果。

## 对新的无标签数据推理

先生成原 V1 基础特征，再使用训练阶段冻结的模板模型补充 V4 特征：

```bash
python src/prepare_inference_features.py \
  --input /path/to/test.parquet \
  --output data/processed/v1_test.parquet \
  --force

python src/prepare_v4_inference_features.py \
  --input /path/to/test.parquet \
  --base-features data/processed/v1_test.parquet \
  --model-dir artifacts/v4_drain_neural/template_model \
  --output data/processed/v4_test.parquet \
  --force

python src/predict_tabular_checkpoint.py \
  --model artifacts/v4_drain_neural/base/model.pt \
  --data data/processed/v4_test.parquet \
  --output artifacts/v4_drain_neural/test_predictions.parquet \
  --device auto \
  --force
```

## 需要回传的结果

训练完成后请提供以下内容：

```bash
cat artifacts/v4_drain_neural/base/metrics.json
cat artifacts/v4_drain_neural/template_model/manifest.json
cat data/processed/v4/v4_manifest.json
tail -n 80 artifacts/v4_drain_neural/base/train_console.log
```

评估时重点比较：

- `competition_score` 和 `macro_f1`；
- benign 被误报为 malicious/suspicious 的数量；
- malicious、suspicious 的召回率；
- `drain_unmatched` 与 `unseen_templates` 数量；
- 训练耗时和模板数量是否过大；
- 是否仍集中错误在 `pipeline=syslog` 且产品为空的混合簇。

README 记录的 V1 无类别权重结果为 `competition_score=0.957210`、
`macro_f1=0.912286`、错误 9,857 行。V3 满分包含第二模型和规则，不应作为本轮
神经网络特征实验的公平基线。
