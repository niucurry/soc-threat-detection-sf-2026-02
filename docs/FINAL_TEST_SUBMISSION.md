# 最终测试集预测与提交

本流程加载已保存的 v5.2-exp01，第 2 轮 checkpoint，不训练、不调阈值、不需要测试标签。
官方要求只提交 UTF-8 CSV `res.csv`，且只有 `event_id,pred_label` 两列。
标签只能是 benign、malicious、suspicious，事件 ID 必须完整覆盖且唯一。
官方样例 res.csv 的标签仅为占位，不能直接提交，也不参与本流程。

## 输入和模型身份

2026-10-14 检查的测试集共 4,234,041 条，ID 全部非空且唯一。原始列为
event_id、timestamp、pipeline、src_ip、dst_ip、src_port、src_host、dst_host、
username、message_sanitized、product_name、vendor_name。

云端需要上传 test_input.parquet，例如放到 `/root/work/test_input.parquet`。
保留原始模型备份和旁边的 `.sha256` 文件，二者不必改名；以下用 BACKUP 变量引用实际文件。
应选择之前保存的 **gap2_complete** 完整包，而非幅度限制较大的对照包。

脚本先校验压缩包 SHA256，再只读取 exp01 的 model.pt，不展开归档中的目录或软链接。
最终权重 SHA256 必须是：

```text
60e2f47a85d0b1f472a8fcf74ea0a8b73df97982817abcd2ebab09141f347080
```

权重中已包含冻结锚点、残差网络和预处理参数，不需要额外加载另一个锚点文件。
脚本严格检查模型为 anchor 输入、content 证据、max_conflict_gap=2、阈值0.5。
其他 checkpoint 即使结构兼容也会被拒绝，以免误用另一次训练的权重。

## 云端运行

在已经配置好的 cm-model 环境中，使用此分支的最新代码。新建独立克隆目录可避免
影响以前的训练目录；已有克隆时在对应目录执行 git pull --ff-only 即可。

```bash
conda activate cm-model
git clone --single-branch --branch release/v5.2-content-rescue https://github.com/niucurry/soc-threat-detection-sf-2026-02.git /root/soc-submit-v5.2
cd /root/soc-submit-v5.2
python -m pip install -r requirements-npu.txt
```

requirements-npu.txt 不重新安装 PyTorch 或 torch-npu；保留与 Ascend 驱动配套的环境。
将 BACKUP 设为你已有的 gap2_complete 压缩包的绝对路径，校验文件必须与它并列。

```bash
TEST=/root/work/test_input.parquet
# 在下一行填入现有完整备份包的真实路径。
BACKUP=/root/work/model_backups/你的完整备份包.tar.gz
ls -lh "$TEST" "$BACKUP" "$BACKUP.sha256"
mkdir -p artifacts/submission_v5_2
nohup bash scripts/predict_test.sh "$TEST" "$BACKUP" artifacts/submission_v5_2 > artifacts/submission_v5_2/nohup.log 2>&1 &
echo $! > artifacts/submission_v5_2/predict.pid
tail -f artifacts/submission_v5_2/nohup.log
```

日志 loaded 阶段显示实际使用的设备；auto 优先 NPU、其次 CUDA、最后 CPU。
如果必须使用 NPU，可在 nohup 前执行 `export PREDICT_DEVICE=npu:0`，设备不可用会报错，
不会悄悄改用 CPU。Ctrl+C 退出 tail 不会停止 nohup 中的预测进程。

完成标记为 `"stage": "complete"`。输出：

```text
artifacts/submission_v5_2/res.csv
artifacts/submission_v5_2/submission_audit.json
```

只下载、提交 res.csv；不要提交分块 parquet、模型、审计 JSON 或原样例文件。
审核命令：

```bash
cat artifacts/submission_v5_2/submission_audit.json
head -n 5 artifacts/submission_v5_2/res.csv
wc -l artifacts/submission_v5_2/res.csv
```

完整官方测试集应有 rows=distinct_event_ids=4234041，invalid_rows=missing_ids=extra_ids=0。
CSV 应有4,234,042行（包括表头）。样本按 event_id 对齐，CSV 行顺序不影响评分。
没有测试集真实标签，因此不能从预测标签数量计算测试分数。

## 推理与恢复的具体行为

每次读取10,000条原始日志。结构特征复用训练 FEATURE_QUERY，UTC 处理保持一致；
内容特征复用 encode_log_content，raw token 长度96、hash桶数读取模型配置。
类别词表、数值归一化和组合计数只读取 checkpoint 的 preprocessor，绝不在测试集 fit。
威胁概率达到0.5才进入威胁分类，子类型从 malicious/suspicious 两类中选择；
不能用三类联合概率的 argmax 替换这个分层判定。

每块推理完成后先写临时 parquet，再原子重命名。只有全部分块完成、CSV 检查通过后，
才发布 res.csv。概率仅保存在中间 parquet，不进入提交文件。
manifest.json 记录输入、权重、源代码指纹和设备/批大小，防止复用其他输入或模型的结果。
断电后用原参数重跑同一命令，完成的分块会显示 reused=true；最多重算中断中的一个块。
不要并行运行同一输出目录，脚本通过 flock 阻止重复写入。
如果输入、代码、设备或批大小改变，必须使用新的输出目录，不能删除 manifest 强行复用。
减小显存占用可设置 PREDICT_BATCH_SIZE=1024；减小内存占用可设置 PREDICT_CHUNK_SIZE=2000。

## 2026-10-14 本地检查范围

- 现存最终备份校验通过，weights_only=True 安全加载，state_dict 严格匹配当前模型。
- 验证集抽取507条，覆盖全部36条原错误、21条修复以及三类代表样本；
  与备份预测相比标签差异0，三类概率最大绝对差约2.3842e-7。
- 测试集抽取256条，分4块完成推理、CSV审计，再启动时4块全部复用，CSV字节一致。
- 测试集全量ID检查通过。没有在本地执行423万条的全量推理，没有重新训练。
- 本地CPU检查不等同于云端NPU已经运行成功。云端须以 complete 日志和审计 JSON 为准。
