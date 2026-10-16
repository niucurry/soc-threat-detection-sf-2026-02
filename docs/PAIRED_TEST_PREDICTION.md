# 在同一份新测试集上比较两套权重

这是现有实验分支的推理工具，不是新的模型版本。无需重新训练。

## 比较对象和限制

| 输出目录 | 权重 | 处理方式 |
|---|---|---|
| old_model | v5.2-exp01 最终内容救援模型 | 加载已核验的完整冻结模型，包含原基础网络和纠错层 |
| new_model | v5.2-exp03-full-label-base | 加载原训练集及原验证标签合并后训练的基础网络，不加载旧纠错层 |

双方各自加载权重内的预处理器，使用同一测试输入，不在测试集上重新拟合任何参数。阈值来自保存的模型。
不使用测试真值，不进行分数搜索，也不复制旧测试集的 73 条人工覆盖。
即使新测试里 event_id 与历史编号相同，也不能把历史标签直接贴上去。

这比较的是两套最终方案，不是只改变训练数据量的严格消融实验，因为它们是否包含纠错层也不同。
只有在同一测试集、相同评分范围下取得的两个分数才能直接比较。
旧的 0.999432 来自另一份测试及人工修改后的输出，不能直接用来判断此次新测试谁更好。
请确认平台允许两次提交、使用原验证标签最终训练，并确认是否采用同样的两列提交格式。
程序只生成文件，不会向平台提交。

## 新 CSV 的处理

必须恰好包含以下 12 个字段，列顺序可不同，但名称不能重复、缺失或多出标签列：

```text
event_id,timestamp,pipeline,src_ip,dst_ip,src_port,src_host,dst_host,username,message_sanitized,product_name,vendor_name
```

`timestamp` 按 float64 Unix 秒读取，不取整、不自动猜测毫秒单位。其他 11 列保留字符串，包括空串、前导零、字面 NA/NULL、引号、换行和 NUL 控制字符。
原 CSV 不改变。转换器输出共同的 Parquet 输入，并检查完整唯一的事件编号、有限数值时间戳及全部记录的列数；遇到错误停止，不跳过错误行。

本次收到的 CSV 实测为 55,837 行、55,837 个唯一事件，SHA256 为：

```text
5a3db3aab57c33680e4627b0df75f9d526445c17964fd9f09d36f44eb6ae1d0b
```

其中 416 行日志含 NUL 字符。Arrow 默认 CSV 读取器在这种输入上发生列数误判，因此使用 Python 标准 CSV 解析器保留原字符，再写 Parquet。
已用独立 DuckDB 读取对照，全部 55,837 行的 12 字段及行序一致。这些数量用于本次人工核对，不硬编码为对其他测试文件的要求。

## 云端命令

先将新的 `test_input.csv` 上传到 `/root/work/test_input.csv`。不要覆盖备份权重。
如果现有项目目录和权重位置仍与前次训练一致：

```bash
conda activate cm-model
cd /root/soc-full-label-refit
git status --short
git switch feat/v5.2-full-label-refit
git pull --ff-only origin feat/v5.2-full-label-refit
ls -lh scripts/predict_model_pair.sh
sha256sum /root/work/test_input.csv
```

如果 Git 提示存在会被覆盖的本地修改，先停止同步、保留修改，不执行 reset/clean。
不需要更新镜像内的 torch、torch_npu 或驱动。依赖与前次训练一致。

检查两套权重及其校验文件，再启动：

```bash
cd /root/soc-full-label-refit
OLD_BACKUP=/root/work/model_backups/v10_gap2_complete_20260904.tar.gz
NEW_MODEL=/root/soc-full-label-refit/artifacts/v5_2_exp03_full_label/final_model/model.pt
OUT=artifacts/test_model_pair_20261016

ls -lh /root/work/test_input.csv "$OLD_BACKUP" "$OLD_BACKUP.sha256" "$NEW_MODEL" "$NEW_MODEL.sha256"
mkdir -p "$OUT"
nohup bash scripts/predict_model_pair.sh /root/work/test_input.csv "$OLD_BACKUP" "$NEW_MODEL" "$OUT" > "$OUT/nohup.log" 2>&1 &
echo $! > "$OUT/predict.pid"
tail -f "$OUT/nohup.log"
```

旧备份保留实际文件名，不要求重命名。旧模型加载时核验备份与冻结权重哈希；新模型核验相邻的 `model.pt.sha256` 及模型格式。
若新权重文件已经不在上述路径，不要拿旧权重替代、不要重训。先找到新模型备份，核验并恢复到另一个目录，然后修改 NEW_MODEL。

默认先运行旧模型，再运行新模型，避免两个进程同时占满一张 NPU。若需要减小设备批量，在首次运行前设置：

```bash
export PREDICT_BATCH_SIZE=1024
```

改变输入、代码、权重、设备或批量配置后使用新的 OUT 目录，不能复用旧分片。
同一配置中断后重跑同一命令会复用已校验的输入和预测分片。Ctrl+C 停止 tail 不会终止后台推理。

## 输出与提交

| 文件 | 用途 |
|---|---|
| input/input_audit.json | 原 CSV 和共同 Parquet 的校验值、行数及字段信息 |
| old_model/res.csv | 旧权重的纯模型预测，提交候选 A |
| new_model/res.csv | 新权重的纯模型预测，提交候选 B |
| old_model/submission_audit.json | 旧预测文件格式、ID、标签数量和校验值 |
| new_model/submission_audit.json | 新预测文件格式、ID、标签数量和校验值 |
| comparison/comparison.json | 两者标签转移数量、变化事件分组、权重和文件身份 |
| comparison/disagreements.csv | 仅包含不同预测的事件、两个标签和原始输入，供人工分析 |

两个提交候选都只有 `event_id,pred_label`，分别保存在不同目录，同名但不能互相覆盖。
比较器按事件编号连接，验证两套推理的输入文件哈希相同，并检查预测文件未在审计后被修改。
差异明细包含日志和额外字段，**不是提交文件**。预测分片中的概率也不加入提交 CSV。

完成后先发送以下输出以便检查：

```bash
cat artifacts/test_model_pair_20261016/old_model/submission_audit.json
cat artifacts/test_model_pair_20261016/new_model/submission_audit.json
cat artifacts/test_model_pair_20261016/comparison/comparison.json
```

保留两份原始预测及对应校验值，分别下载并在平台允许的情况下提交；记录每个文件对应的分数和评分范围。
不要仅凭标签分歧决定手工改标签，也不要把比较器中的空 competition_score 当作运行失败：没有真值就不会计算准确率或竞赛分数。

## 功能验证

2026-10-16：本地完整测试 73 项通过，其中新增 19 项；Bash 语法检查通过。
新增测试覆盖 CSV 字符/空值/时间戳保留、字段与 ID 异常拒绝、输入缓存失效、训练推理特征一致性、乱序预测对齐、旧输入与被改动输出拒绝。
原冻结权重在本次新测试抽取的 3 行上完成 CPU 推理检查，包括一条含 NUL 的日志。
这些是工程检查，不是两套权重在新测试集的全量 NPU 或官方评分结果。
