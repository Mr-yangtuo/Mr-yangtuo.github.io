# 睁闭眼识别：嵌套调参对照报告

[打开在线报告](https://mr-yangtuo.github.io/eeg-eye-state/)

2026-11-17 更新。53 人、3,681 个两秒窗口、原按人五折保持不变。报告有10个页签，保留此前固定参数方法和失败分析。

| 方法 | 原固定参数 | 仅调参数 | 再调阈值 | 调阈值后平衡准确率 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 逻辑回归 | 82.23% | 81.88% | 81.83% | 82.66% | 0.8911 |
| RBF SVM | 82.72% | 82.56% | 82.40% | 83.19% | 0.8954 |
| 随机森林 | 81.77% | 81.58% | 81.77% | 82.53% | 0.8861 |
| 梯度提升树 | 82.12% | 81.64% | 80.87% | 81.86% | 0.8857 |
| SVM + 特征选择 | 82.72% | 82.97% | 83.02% | 83.80% | 0.8963 |

每个外层训练集内再做按人四折，选参目标是每人等权平衡准确率。搜索正则化、非线性核、树容量、训练权重，再以内部折外预测单独选择阈值。SVM采用对数范围随机搜索与前两名附近局部细化。特征选择实验另列，在内部选择全部20维、相对功率10维或log绝对功率10维。

外层评分没有进入本轮参数选择；但搜索空间设计参考了过去对同一批受试者的分析，结果仍是探索性比较。不能把内部最高分当作新人的测试表现。

## 复现

公开内容为汇报、汇总指标与代码。课程 EEG 和量表留在本地。将 code 内文件放入原项目的 eye_state_baseline 目录，原项目根目录需包含 data_processed 与 subjects_summary.csv。

```powershell
python -m pip install -r eye_state_baseline/requirements.txt
python eye_state_baseline/run_baseline.py
python eye_state_baseline/build_report.py
python eye_state_baseline/compare_models.py --stage 1
python eye_state_baseline/compare_models.py --stage 2
python eye_state_baseline/tune_models.py --workers 8
python eye_state_baseline/verify_tuning.py
python eye_state_baseline/build_comparison.py
```

tuning 目录按模型和外折保存所有搜索分数、内外层人员名单、内层分数、测试预测、阈值以及拟合模型。脚本可以从已完成折恢复。本次运行使用Python3.14，画图字体默认Windows微软雅黑。

## 结果口径

原固定参数、选参默认阈值、选参后调阈值都报告。概率模型默认阈值0.5；SVM默认决策分数0，SVM得分未经概率校准。四个基础模型保持20维，SVM特征选择另作输入变化实验。没有删除低分受试者或改变标签。

第一轮Stacking仍保留为参照，未重做其多层调参。最终按人配对重采样区间基于固定外层预测，没有重训模型，不包括方案选择偏差。报告内保留所有方法的具体说明、参数、失败案例与优化方向。
