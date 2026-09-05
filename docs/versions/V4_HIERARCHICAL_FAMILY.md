# V4：分层威胁检测与子类型分类

V4 先判断 `benign/threat`，只有进入 threat 后再判断 `malicious/suspicious`。最终概率为：

```text
P(benign)     = P(benign)
P(malicious)  = P(threat) × P(malicious | threat)
P(suspicious) = P(threat) × P(suspicious | threat)
```

## v4.0 输入和模型

Metadata tower 输入 9 个类别字段：v1.0 的 8 个类别字段加 `vendor_name`；输入 v1.0 全部
23 个数值字段。Semantic tower 输入 4 个类别字段：`content_family`、`content_action`、
`content_protocol`、`content_event_code`；以及 4 个数值字段：`content_has_threat`、
`content_has_authentication`、`content_has_potentially_harmful`、`raw_token_count`。
Content tower 输入 v3.0 的 96 个 `raw_token_ids`。

三塔分别形成 128、64、128 维向量。Threat head 融合三塔；另设 metadata/content threat
辅助头。Subtype head 以 metadata 输出为基础，只让 semantic/content residual 受控修正。
训练损失为 threat CE + 0.75 subtype CE + 0.15 metadata-threat CE + 0.25 content-threat CE
+ 0.35 metadata-subtype CE。

`exp01` 不使用新颖度门控。`exp02` 统计训练输入中
`product_name + content_family + content_action` 的出现次数，门控为 `count/(count+32)`；
这个统计不读取标签和验证集。

两组历史正式结果都为 Score `0.9993140394`、46 错、0 FP、46 FN；exp02 Log Loss
`0.0002620398` 优于 exp01 的 `0.0003264483`，但 hard prediction 相同。因此主要收益来自
分层任务，不是门控。全局 threat threshold 从 0.5 降到 0.2 仍是相同 46 个漏报，降到
0.02 只修 2 个 FN 却增加 97 个 FP，排除了“只调阈值”的做法。

原 46 错 checkpoint 后来丢失，不能用指标 JSON 冒充权重。相同数据和超参的 seed29 恢复
checkpoint 是当前可用锚点：Score `0.9993387402`、57 个三分类错误，其中 FP=1、FN=32、
子类型混淆=24。后续残差实验必须以 seed29 自己的 epoch-0 输出为严格对照；原 46 错结果
只能作为历史参考。

入口：`scripts/run_cloud_v4_0_hierarchical.sh`；种子复现：
`scripts/run_cloud_v4_0_seed_sweep.sh`。

## v4.1 多视图和证据保持

多视图把长正文分成 head、middle、tail、key_value 四组，每组 64 token。key_value 视图完整
优先字段集合为：action、act、status、result、reason、outcome、decision、event、eventid、
event_id、eventcode、event_code、process、processname、process_name、command、commandline、
command_line、protocol、severity、category、operation。

三个实验：标准多视图 208 错/Score `0.9967185764`；原内容加证据保持 56 错/
`0.9992727909`；多视图加证据保持 66 错/`0.9992315501`。均未超过 v4.0。多视图从头
替换表示会破坏已经学会的动作语义，证据保持能恢复大部分退化但制造新的正常误报。结论是
下一版必须冻结可靠模型，只允许零初始化、可回退的局部残差。

入口：`scripts/run_cloud_v4_1_multiview.sh`。
