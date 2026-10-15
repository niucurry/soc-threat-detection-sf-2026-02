# v5.2-exp03：全部标签重训基础网络

本实验位于 `feat/v5.2-full-label-refit`，是现有模型代码上的重训对照，不新增主版本。
它重新训练分层神经网络的全部参数。**不加载旧基础权重，不沿用旧纠错层，也不自动覆盖之前人工修改的 73 条标签。**
因此，输出名称为 `v5.2-exp03-full-label-base`。它是候选模型，不代表已经超过现有提交。

## 为什么做这个实验

现有最终模型的基础网络被冻结，后续纠错层只能有限度地提高威胁分数。
仅把新增数据送给旧纠错脚本，不会使基础网络学会新增产品、正文模式或威胁子类型。
此次检验的假设是：将原验证集中的正确标签作为额外监督，能否改善基础网络。
新增数据不是测试集，也不能保证包含测试集的全部情况。

已知提交基准为用户反馈的 **0.999432**，对应人工修改 73 条之后的文件。
没有未修改文件的官方分数、分项指标或测试集真实标签。
该基准只用于最终提交效果比较，不是“只增加训练数据”的严格消融对照。
还需确认赛事允许使用已提供的验证标签做最终训练；格式说明没有明确训练数据权限。

## 数据与标签如何合并

只接受 `train.parquet`、`valid_input.parquet`、`valid_answer_private.parquet`。
准备脚本没有测试集参数，不能把提交预测当成真实标签加入训练。

1. 分别检查训练输入、验证输入的 ID 完整且唯一。
2. 检查验证答案 ID 完整且唯一，且恰好覆盖验证输入，标签只能为三个合法类别。
3. 验证输入只与自己的答案按 ID 对齐，绝不拿测试 ID 对接历史答案。
4. 内部样本 ID 加来源前缀，例如 `train:EVT-0000000001`、`valid:EVT-0000000001`。
   原 ID 保存在 `original_event_id`，来源保存在 `source_split`，均不作为模型输入。
5. 最终训练文件保留两个来源的全部记录，不按跨来源的裸 ID 去重。

按已有数据统计，合计应为 **4,070,923 行**，其中 benign 3,859,296、malicious 125,780、suspicious 85,847。
实际行数和类别数以本次 `features/split_report.json` 为准，脚本不把这些历史数量硬编码成标签。

## 内部对照怎样划分

先为记录生成仅用于划分的 `group_key`：

- 非空日志：pipeline、vendor、product 加正文。正文折叠空白，替换明确的年月日时分秒时间字符串。
- 保留 IP、用户、端口、路径、动作和事件码等差异，不把所有日志归并为模板。
- 空日志：使用除 event_id、timestamp 外的全部输入字段，NULL 与空串统一。
- 相同分组不会同时进入开发训练和留出集。

这是可复现的重复/部分近重复隔离方法，**不保证发现所有近重复，也不是时间外推实验**。
空日志可能形成大组，需检查报告中的最大组大小及实际留出行比例。

从原验证集的分组中，按组内完整标签集合分层，固定随机种子抽取约 20% 的组作为开发留出。
每个分层至少保留一个训练组；只有一个组的分层不进入留出。
20% 是组比例，**不是固定 20% 的行数**。若任何训练/留出文件缺少一个类别，脚本报错，不悄悄更换划分。

| 文件 | 内容 | 用途 |
|---|---|---|
| control_train.parquet | 原训练集剔除与留出组重叠的记录 | 对照训练 |
| expanded_train.parquet | 对照训练记录加原验证集未留出的记录 | 扩充训练 |
| holdout.parquet | 原验证集中选定组的记录 | 两组共同的开发评价 |
| full_train.parquet | 原训练集及带正确标签的全部原验证集 | 最终固定轮次重训 |

组内如果出现不同标签，不拆组、不静默改标签，报告冲突组数量。
当前原验证集已经被用于历史开发，因此新划分只能用于开发比较，不能声称是从未接触过的盲测。
特征提取本身是固定哈希和逐行规则，无需拟合。类别编码、数值缩放及组合频次只在每次训练的训练部分拟合。

## 模型与训练方式

完整输入字段沿用本分支 README 的基础网络输入：元数据 9 类别＋23 数值、语义 4 类别＋4 数值，以及 96 位 `raw_token_ids`。
不存在新增标签特征。`group_key`、来源、原始 ID 和时间划分辅助信息不进入模型。
基础网络先识别 benign/threat，再在 threat 内区分 malicious/suspicious，频次门控方式为 count。

训练设置：65536 哈希桶；内容 embedding 64、输出 128；token dropout 0.05、category dropout 0.02；
AdamW，学习率 0.0007，weight decay 0.00001；batch 2048，评价 batch 4096，梯度裁剪 5。
威胁类别权重指数 0.25，子类型权重指数 0。
损失为威胁 CE＋0.75 子类型 CE＋0.15 元数据威胁辅助 CE＋0.25 内容威胁辅助 CE＋0.35 元数据子类型辅助 CE。
威胁阈值固定 0.5，不根据测试得分搜索阈值。本实验不加入证据保持损失或残差纠错层。

默认 3 个预先固定的种子：20261015、20261016、20261017。每个种子跑对照和扩充两组，共 6 次。
两组采用相同配置，但数据量不同，因此同一轮数不代表完全相同的梯度更新次数。
每组最多 10 轮、patience 4，按开发分数、内容分支 F1、log loss 顺序选模型。
汇总均值、标准差、最差/最好结果、FP、FN、子类型混淆和每个种子的配对差异，不达到某个高分就提前停止种子实验。

最终模型使用预先声明的第一个种子，轮数取三个扩充实验最佳轮次的中位数。
它从头训练全部标签，不读取验证集、不早停、不输出假冒的验证分数。
全量训练 `metrics.json` 中 `competition_score` 为 null，这是正常情况。
若扩充组平均分没有改善，默认拒绝全量重训。需要先分析结果，明确接受风险后才使用 `REFIT_ALLOW_NONIMPROVING=1`。

## 云端操作

建议新建克隆目录，避免影响正在使用的最终提交目录。
不要重新安装镜像中的 torch、torch_npu 或驱动。

```bash
conda activate cm-model
cd /root
git clone --single-branch --branch feat/v5.2-full-label-refit https://github.com/niucurry/soc-threat-detection-sf-2026-02.git soc-full-label-refit
cd /root/soc-full-label-refit
python -m pip install -r requirements-npu.txt
ls -lh /root/work/train.parquet /root/work/valid_input.parquet /root/work/valid_answer_private.parquet
```

### 第一步：数据准备和内部对照

```bash
mkdir -p artifacts/v5_2_exp03_full_label
nohup bash scripts/run_full_label_refit.sh study /root/work artifacts/v5_2_exp03_full_label > artifacts/v5_2_exp03_full_label/study.log 2>&1 &
echo $! > artifacts/v5_2_exp03_full_label/study.pid
tail -f artifacts/v5_2_exp03_full_label/study.log
```

结束后提供这两个文件的内容，先分析是否继续：

```bash
cat artifacts/v5_2_exp03_full_label/features/split_report.json
cat artifacts/v5_2_exp03_full_label/study_comparison.json
```

若初次需要减小设备批量，在第一次启动前设置 `export REFIT_BATCH_SIZE=1024`。
改变配置后应使用新的输出目录，不能在旧目录里混用。
建议主存至少 16GB，条件允许优先 32GB；大数据训练会整体载入紧凑特征，不是完全流式训练。实际峰值需云端观测。
减小 batch 主要减少设备内存占用，不能解决整体载入特征导致的主存不足。

### 第二步：确认方案后全量重训

```bash
nohup bash scripts/run_full_label_refit.sh refit /root/work artifacts/v5_2_exp03_full_label > artifacts/v5_2_exp03_full_label/refit.log 2>&1 &
echo $! > artifacts/v5_2_exp03_full_label/refit.pid
tail -f artifacts/v5_2_exp03_full_label/refit.log
```

该阶段读取第一步保存的计划。不要手动将原验证文件再传给训练入口当验证集。
成功后权重位于 `artifacts/v5_2_exp03_full_label/final_model/model.pt`，并生成同目录校验文件。
工作流根目录自动生成 `v5_2_exp03_full_label_model.tar.gz` 及 `.sha256`，包含权重、预处理、计划、数据校验信息、实验结果、代码快照、提交号和运行环境。
请把这两个备份文件下载到本地；云端同盘副本不防机器或磁盘丢失。

### 第三步：生成测试预测文件

```bash
mkdir -p artifacts/submission_full_label
nohup bash scripts/predict_full_label_refit.sh /root/work/test_input.parquet artifacts/v5_2_exp03_full_label/final_model/model.pt artifacts/submission_full_label > artifacts/submission_full_label/nohup.log 2>&1 &
echo $! > artifacts/submission_full_label/predict.pid
tail -f artifacts/submission_full_label/nohup.log
```

提交候选文件为 `artifacts/submission_full_label/res.csv`。
该入口先验证新权重的 SHA256，再预测。旧 `scripts/predict_test.sh` 继续只接受原来已经核验的最终权重，不能混用。
预测端与训练端共享同一逐行特征函数，只输出两列并验证全量 ID、唯一性、合法标签和缺失/多余事件。
程序不提交文件到平台。测试准确性仍需平台评分，格式通过不等于标签全部正确。

## 中断恢复和可复现边界

- 数据准备按 20000 行分片，写完后原子发布并保存校验值；再次运行同一命令会复用通过校验的分片。
- 每个训练轮次结束后保存 `last.pt`，包含参数、优化器、最佳状态、历史和随机状态。
  中断后重跑同一命令，从最后完成的轮次继续；中途未完成的一个轮次会重跑。
- 六个开发实验中已完整完成且文件校验一致的实验会跳过。
- 测试预测同样按分片续跑。不同输入、代码或设置不能复用旧输出目录。
- 脚本使用 `flock` 阻止同一输出目录重复启动。按 Ctrl+C 停止 `tail -f` 不会停止后台任务。
- 不承诺跨硬件、驱动或不同 NPU 算子实现逐位复现。保存的实验结果及文件哈希才是核验依据。

## 测试结果与待验证事项

本地自动测试覆盖来源 ID 隔离、验证答案完整性、分组泄漏、全部标签保留、缓存失效、固定轮次训练、
中断恢复、训练与预测特征/结果一致性、命令行端到端预测及提交审计。
2026-10-15 本地 `python -m pytest -q`：**54 passed**，包含 11 项新增测试；Bash 语法检查通过。
CPU 模拟轮次边界中断，恢复后的参数与不中断训练逐位一致；这不代表已验证 NPU 逐位一致。
另外用原冻结权重抽查 3 条历史预测，标签差异为 0；这不是旧模型全量回归评测。
这些是合成小样本上的功能测试，不是云端 NPU 全量训练成绩。
本实验尚无新增的完整验证/官方测试分数，不能把历史 0.999432 写成重训结果。
后续是否为新基础网络训练纠错层，需要看新错误分布，并单独做同一开发留出上的对照。
