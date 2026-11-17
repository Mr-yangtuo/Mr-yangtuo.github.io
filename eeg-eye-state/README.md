# 睁闭眼识别：五模型对照实验

[打开在线报告](https://mr-yangtuo.github.io/eeg-eye-state/)

2026-11-17 更新。53 名受试者、3,681 个两秒窗口、20 维频域特征，所有模型沿用相同的按人分组五折。原基线逐窗预测已复现。

| 方法 | 准确率 | 宏平均 F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| 逻辑回归 | 82.23% | 0.8199 | 0.8886 |
| RBF SVM | 82.72% | 0.8253 | 0.8910 |
| 随机森林 | 81.77% | 0.8153 | 0.8862 |
| 梯度提升树 | 82.12% | 0.8186 | 0.8869 |
| 分组 Stacking | 82.64% | 0.8245 | 0.8913 |

各新方法相对逻辑回归的配对按人重采样差值区间均跨零，尚不能确认稳定提升。第二阶段方案参考了第一阶段结果，因此是探索性方法比较，需新增独立受试者验证。

报告包含数据说明、共同控制条件、每种方法的动机/预期/流程/参数/结果/可能原因、失败案例检查器、全方法对照和优化路线。点击页签切换；图表已内嵌，下载 index.html 后也可离线查看。

## 复现

公开目录包含汇报、汇总指标和代码。复现需要课程提供的原始项目数据；脑电记录和量表不随本仓库分发。

在原课程项目文件夹创建 `eye_state_baseline`，将 `code/` 中的 Python 脚本和 requirements.txt 放入该目录。项目根目录需有 `data_processed/` 与 `subjects_summary.csv`。支持 Python 3.14；绘图默认使用 Windows 微软雅黑字体，可在代码中更换字体路径。

在项目根目录执行：

```powershell
python -m pip install -r eye_state_baseline/requirements.txt
python eye_state_baseline/run_baseline.py
python eye_state_baseline/build_report.py
python eye_state_baseline/compare_models.py --stage 1
python eye_state_baseline/compare_models.py --stage 2
python eye_state_baseline/verify_comparison.py
python eye_state_baseline/build_comparison.py
```

`results/` 保存基线数据、特征和预测。`comparison/` 保存五模型逐窗折外预测、内外层人员划分、参数、配对比较、输入哈希和核验结果。

上游数据已经过课程方滤波与去眼电，本实验在其基础上筛选、分窗、去均值和提取特征。标签 0 为闭眼、1 为睁眼；标签 3 视频任务不参与模型。

SVM 使用决策分数计算 AUC，分数未校准为概率。Stacking 内部使用按人四折生成元训练数据，外层沿用原按人五折。全部比较不进行超参数搜索、不删除低分人员，不把测试人员数据用于标准化或融合器训练。
