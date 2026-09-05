# V2：神经网络基础模型 + 路由正文专模 + 规则

V2 是双模型混合族，不是单个神经网络。第一模型为 v1.0 表格 MLP，覆盖全部日志；第二模型
只处理 `pipeline=syslog AND product_name 为空` 的混合簇。这个路由在训练集有 342,820 行，
验证集有 301,333 行，只包含 benign/malicious，因此第二模型做二分类。

正文专模使用全部 `message_sanitized`：小写 word unigram/bigram TF-IDF，`min_df=2`、
`max_df=0.9999`、最多 200,000 维、sublinear TF；分类器是 L2 正则、平均权重的
`SGDClassifier(loss="log_loss")`。高置信语义层识别 `REJECT OK`、Windows 4625、明确 deny/
drop、block-url 等模式。路由外保留第一模型结果，路由内用正文专模修正 benign/malicious；
可疑类仍由第一模型负责。

## v2.0

形成上述双模型、路由和保守/调优两种输出。优点是精准修复 v1.0 的歧义簇，不扰动其余
约 171 万行；缺点是阈值最初按专模二分类 Macro-F1 选，不能代表完整比赛指标。

## v2.1

架构不变，只把第一模型 checkpoint 和正文阈值都改为按完整比赛公式选择。公式权重为：
Threat Binary F1 0.40、Threat Binary Recall 0.25、两类威胁 Recall 平均 0.15、
Macro-F1 0.10、Soft Label Score 0.05、Balanced Accuracy 0.05。

保守输出在完整验证集 Score `0.9998902649`、Macro-F1 `0.9999579178`、10 个
`suspicious -> benign`；调优输出加 10 条验证错误启发的可疑覆盖后为 1.0。满分不能作为
独立泛化证据，因为 Symantec DLP 和 Duo 覆盖条件是在反复查看这份验证集后加入的。

## v2.2

保持 v2.1 架构，增加家族隔离、字符/词符对照、概率间隙和规则纯度审计。字符模型、
无标签自适应阈值等实验没有稳定收益，因此正式分类结果仍为保守 10 错、调优 0 错。
v2.2 是很强的工程基线，但包含第二模型和规则，不能和“单神经网络是否真正学会内容”混为
同一研究结论。

完整入口：`scripts/run_cloud_v2_2_hybrid.sh`。只刷新正文专模和语义层：
`scripts/run_cloud_v2_2_text_refresh.sh`。
