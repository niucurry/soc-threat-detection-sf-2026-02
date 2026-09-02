# V8 多视图内容与证据保留实验

## 1. 实验依据

V7-H2 在完整验证集上只有 46 个错误，全部是威胁被判为 benign：

```text
36 malicious -> benign
10 suspicious -> benign
0 benign -> threat
0 malicious <-> suspicious
```

阈值从 0.5 降到 0.2 时结果完全不变；降到 0.02 只修复 2 条漏报，
却引入 97 条正常误报。这说明剩余问题不是阈值，而是输入覆盖和威胁融合。

46 条错误的 `raw_token_count` 全部等于 V7 上限 96，45 条正文长度超过
1000 字符。另有 6 条 malicious 日志的元数据威胁概率约为 0.957，最终威胁
概率却只有 0.011～0.022，证明融合头可能消除单个辅助分支的有效证据。

## 2. 四视图内容编码

V8 不使用 Drain、模板 ID、schema ID、事件 ID 或验证标签。每条正文生成四个
独立视图，每个视图默认 64 个固定哈希 token：

```text
head      正文开头字符区域中的词、bigram 和字符 n-gram
middle    正文中部字符区域中的词、bigram 和字符 n-gram
tail      正文结尾字符区域中的词、bigram 和字符 n-gram
key_value JSON、XML、key=value 中的字段、值和字段=值关系
```

IP、UUID、时间戳、长数字、长十六进制串和脱敏实体继续归一化。键值视图保留
如下关系，而不是只保留字段名：

```text
"result":"invalid_passcode"
  -> kv_key:result
  -> kv_value:invalid_passcode
  -> kv_pair:result=invalid_passcode
```

动作、结果、原因、事件码、进程、命令、协议和严重度等通用安全字段优先进入
键值视图；其他字段从开头、中部和结尾均匀选择。这个优先级不读取标签，也不
针对已知验证错误写厂商或事件规则。

四个视图使用共享 Embedding 和共享内容编码器，随后加入可学习的视图位置向量，
拼接后由多层感知机融合。默认总宽度为 `4 * 64 = 256`。

## 3. 证据保留损失

V7 已有最终威胁头、元数据威胁辅助头和内容威胁辅助头。V8-B/V8-C 对训练中的
真实威胁样本增加：

```text
final_margin    = final_threat_logit - final_benign_logit
metadata_margin = metadata_threat_logit - metadata_benign_logit
content_margin  = content_threat_logit - content_benign_logit

target_margin = max(
    positive_floor,
    max(metadata_margin, content_margin) - allowed_gap
)

evidence_loss = relu(target_margin - final_margin)
```

辅助分支 margin 在该损失中会停止梯度，避免模型通过主动削弱辅助证据来降低
约束。该损失只作用于训练标签为 malicious/suspicious 的行；benign 行仍由原
威胁交叉熵约束，所以不是“看到 deny 就强制报威胁”的推理规则。

默认设置：

```text
evidence_preservation_weight = 0.20
positive_threat_margin = 0.0
allowed_branch_logit_gap = 0.5
```

## 4. 三组严格对照

| 运行目录 | 内容输入 | 证据保留 | 要回答的问题 |
|---|---|---:|---|
| `a1_multiview_standard` | 四视图 | 0 | 只扩大正文覆盖是否有效 |
| `b1_raw_evidence` | V7原96-token | 0.20 | 只修复融合压制是否有效 |
| `c1_multiview_evidence` | 四视图 | 0.20 | 两项改动是否互补 |

三组都保留 V7-H2 的分层分类、训练集组合频次门控、类别权重和其他损失参数。
脚本先训练 B，因为它能直接复用 V6 特征；随后才生成可恢复的 V8 分片并训练
A/C。云平台中断后再次运行同一命令，完整分片和已经产生 `metrics.json` 的实验
都会自动跳过。

## 5. 云平台运行

```bash
git fetch origin
git switch --track origin/feat/v8-multiview-evidence-fusion
git pull --ff-only

mkdir -p artifacts/v8_multiview_evidence
nohup bash scripts/run_cloud_v8_multiview_evidence.sh /root/work \
  > artifacts/v8_multiview_evidence/nohup.log 2>&1 &
echo $! > artifacts/v8_multiview_evidence/train.pid
tail -f artifacts/v8_multiview_evidence/nohup.log
```

若 NPU 显存不足：

```bash
nohup env V8_BATCH_SIZE=1024 V8_VALID_BATCH_SIZE=2048 \
  bash scripts/run_cloud_v8_multiview_evidence.sh /root/work \
  > artifacts/v8_multiview_evidence/nohup.log 2>&1 &
```

恢复中断任务时不要设置 `V8_FORCE_PREPARE=1` 或 `V8_FORCE_TRAIN=1`。
只有主动修改 token 参数后才强制重建 V8 分片。

## 6. 输出与验收

```bash
cat artifacts/v8_multiview_evidence/comparison.json
cat data/processed/v8/v8_manifest.json

for run in b1_raw_evidence a1_multiview_standard c1_multiview_evidence; do
  cat "artifacts/v8_multiview_evidence/${run}/metrics.json"
  cat "artifacts/v8_multiview_evidence/${run}/analysis/error_summary.json"
done

cat artifacts/v8_multiview_evidence/a1_multiview_standard/analysis/content_coverage.json
cat artifacts/v8_multiview_evidence/c1_multiview_evidence/analysis/content_coverage.json
```

主要验收条件：

```text
competition_score > 0.9993140394
errors < 46
threat_false_positive 尽量保持在 0～2
threat_false_negative < 46
subtype_confusion = 0
v7_h2_comparison.new_in_candidate 尽量为 0
```

## 7. 当前验证状态

- V8专项测试、原内容测试和分层模型测试已通过；
- 使用3000行真实Parquet完成V6特征、V8分片、事件ID连接和三组CPU训练；
- 模型、指标、预测、错误CSV和内容覆盖审计均成功生成；
- 上述小样本只验证代码链路，不作为正式精度结论；
- V8完整云端结果尚未产生，不能提前声称优于V7。
