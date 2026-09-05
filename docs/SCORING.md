# 评分公式与结果口径

标签顺序固定为benign、malicious、suspicious。C[i,j]表示真实类别i、预测类别j的样本数，
N为全体样本数。threat合并malicious与suspicious。

TP=C[m,m]+C[m,s]+C[s,m]+C[s,s]；
FP=C[b,m]+C[b,s]；FN=C[m,b]+C[s,b]。
Threat precision=TP/(TP+FP)，Threat recall=TP/(TP+FN)，
Threat F1=2TP/(2TP+FP+FN)。分母为0时按实现返回0。

单类precision=C[i,i]/列和，recall=C[i,i]/行和，
F1=2*precision*recall/(precision+recall)；
Macro-F1为三类F1的算术平均，Balanced Accuracy为三类recall的算术平均。
Threat Recall为malicious recall与suspicious recall的算术平均，
与按样本数加权的二分类Threat recall不同。

Soft Label Score=(完全正确行数+0.5*(C[m,s]+C[s,m]))/N。
这是项目依据比赛评分说明采用的逐行平均口径，结果文件保存该定义。
正常/威胁互错计0；威胁内部子类型互错计0.5。

```text
Competition Score =
  0.40 * Threat Binary F1
+ 0.25 * Threat Binary Recall
+ 0.15 * Threat Recall
+ 0.10 * Macro-F1
+ 0.05 * Soft Label Score
+ 0.05 * Balanced Accuracy
```

Accuracy=trace(C)/N。三分类错误=N-trace(C)=FP+FN+子类型互错。
Log Loss=-mean(log(P[真实类别]))，由三类概率计算；不能仅从混淆矩阵得到。
hard分类相同可能有不同Log Loss；较低Log Loss不等于已证明概率校准更好。

实际例子：v4.0 seed20260828有50错（FP0/FN50/子类型0），
seed20260829有57错（FP1/FN32/子类型24）。
后者减少18个漏报，增加1个FP及24个子类型互错，Score从0.9992500022升到0.9993387402。
所以少错不总等于比赛分高；模型选择须一起报告Score、FP、FN、子类型互错和Log Loss。

分层模型先以P(threat)≥阈值决定threat，再在两类子类型中取最大概率；
这与直接对三个乘积概率argmax可能不同。正式指标必须使用训练程序相同的决策方式。
最终概率为P(b)=1-P(t)，P(m)=P(t)*P(m|t)，P(s)=P(t)*P(s|t)。

残差训练从epoch0锚点开始选择，优先Score，其次三分类错误更少、再其次Log Loss更低；
只保证在被选择的这份验证指标上不劣，不能保证未知数据不劣。
路由内部301,333行和全量2,014,052行的指标必须明确区分；小样本烟雾测试只验证代码链路。
