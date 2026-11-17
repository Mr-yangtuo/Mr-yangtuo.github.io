"""Build offline tabbed report; publish only the resulting report and source code."""
from pathlib import Path
import json,re,html,base64,io,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

B=Path(__file__).resolve().parent; O=B/'comparison'; F=O/'figures'; F.mkdir(exist_ok=True)
M=json.loads((O/'metrics.json').read_text()); A=json.loads((O/'analysis.json').read_text()); P=json.loads((O/'protocol.json').read_text())
NAMES={'lr':'逻辑回归','svm':'RBF SVM','rf':'随机森林','hgb':'梯度提升树','stack':'Stacking 集成'}
COLORS={'lr':'#078b89','svm':'#346cac','rf':'#b78538','hgb':'#9c5a7d','stack':'#536cb7'}
ORDER=list(NAMES); DIAG={d['subject']:d for d in A['diagnostics']}
PREDS={m:pd.read_csv(O/(m+'_oof.csv')) for m in ORDER}
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':11,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cbd8dd',
    'text.color':'#182d3d','axes.labelcolor':'#526873','xtick.color':'#526873','ytick.color':'#526873','svg.fonttype':'path'})
def pc(x,n=1):return f'{x*100:.{n}f}%'
def pp(x):return f'{x*100:+.2f}'
def image(name,fig,alt):
    fig.savefig(F/(name+'.svg'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight')
    buf=io.BytesIO();fig.savefig(buf,format='svg',bbox_inches='tight');plt.close(fig)
    return '<img class="chart" alt="'+alt+'" src="data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
def heading(kicker,title,sub=''):
    return f'<div class="intro"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p class="lead">{sub}</p></div>'
def statrow(values):return '<div class="resultmetrics">'+''.join(f'<div><strong>{v}</strong><span>{k}</span></div>' for k,v in values)+'</div>'
def method_plot(m):
    data=M[m];fig,(a,b)=plt.subplots(1,2,figsize=(10.8,3.5),gridspec_kw={'width_ratios':[.85,1.35]},layout='constrained')
    cm=np.array(data['cm']);a.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='BuGn')
    for i in range(2):
        for j in range(2):
            v=cm[i,j]/cm[i].sum();a.text(j,i,f'{cm[i,j]:,}\n{pc(v)}',ha='center',va='center',fontsize=14,color='white' if v>.6 else '#182d3d')
    a.set_xticks([0,1],['预测闭眼','预测睁眼']);a.set_yticks([0,1],['真实闭眼','真实睁眼']);a.set_title('混淆矩阵 · 个数 / 行占比',fontsize=12,pad=15)
    pos=np.arange(1,6);b.plot(pos,[r['train_accuracy']*100 for r in data['folds']],'o--',label='训练（诊断用）',color='#b1b9bd',lw=1.5)
    b.plot(pos,[r['accuracy']*100 for r in data['folds']],'o-',label='测试（评价用）',color=COLORS[m],lw=2)
    for x,r in zip(pos,data['folds']):b.annotate(pc(r['accuracy']),(x,r['accuracy']*100),xytext=(0,-19),textcoords='offset points',ha='center',fontsize=10)
    b.set_xticks(pos,[f'第 {i} 折' for i in pos]);b.set_ylim(65,102);b.set_ylabel('准确率（%）');b.grid(alpha=.15);b.legend(frameon=False,ncol=2,loc='upper center');b.set_title('相同五组人员，轮流测试',fontsize=12,pad=15)
    return image(m+'_evaluation',fig,NAMES[m]+'混淆矩阵及每折训练和测试准确率')

def overview_plots():
    out={}
    fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained')
    for i,m in enumerate(ORDER):
        v=M[m]['accuracy']*100;lo,hi=np.array(M[m]['ci95'])*100
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[m],capsize=4,ms=7)
        ax.text(89,i,f'{v:.2f}%',va='center',color=COLORS[m],fontsize=11)
    ax.set_yticks(range(5),list(NAMES.values()));ax.invert_yaxis();ax.set_xlim(73,95);ax.set_xlabel('准确率与按人重采样 95% 区间（%）');ax.grid(axis='x',alpha=.15)
    out['accuracy']=image('accuracy_compare',fig,'五种模型准确率与按人重采样区间')
    fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained')
    for i,v in enumerate(A['paired_vs_lr']):
        x=v['delta_accuracy']*100;lo,hi=np.array(v['ci95'])*100
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=COLORS[v['method']],capsize=4,ms=7)
        ax.annotate(f'{x:+.2f}',(x,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
    ax.axvline(0,c='#9dabab',ls='--');ax.set_yticks(range(4),[NAMES[v['method']] for v in A['paired_vs_lr']]);ax.invert_yaxis();ax.set_xlim(-2,2);ax.set_xlabel('相对逻辑回归的准确率变化（百分点）');ax.grid(axis='x',alpha=.15)
    out['delta']=image('paired_delta',fig,'相对逻辑回归准确率差的配对按人重采样区间')
    frame=pd.read_csv(O/'subject_comparison.csv').set_index('subject').sort_values('lr').head(12)
    fig,ax=plt.subplots(figsize=(7.8,5.8),layout='constrained')
    ax.imshow(frame.to_numpy()*100,cmap=LinearSegmentedColormap.from_list('scores',['#fae0d7','#f3ede0','#b3dcd4','#178b83']),vmin=0,vmax=100,aspect='auto')
    for i in range(len(frame)):
        for j in range(5):ax.text(j,i,f'{frame.iloc[i,j]*100:.1f}%',va='center',ha='center',fontsize=10,color='#18383f')
    ax.set_xticks(range(5),list(NAMES.values()));ax.set_yticks(range(len(frame)),frame.index);ax.tick_params(length=0)
    out['heatmap']=image('hard_subjects',fig,'按原基线准确率排序最差12人的五模型准确率对比')
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for d in A['diagnostics']:ax.scatter(d['alpha_gap']*100,d['accuracy']['lr']*100,s=25,c='#9ac4c2',alpha=.8)
    for s in ['S0013','S0037','S0043']:
        d=DIAG[s];ax.scatter(d['alpha_gap']*100,d['accuracy']['lr']*100,c='#c17836',s=42);ax.annotate(s,(d['alpha_gap']*100,d['accuracy']['lr']*100),xytext=(7,-4),textcoords='offset points',fontsize=10)
    ax.axvline(0,c='#b5c1c5',ls='--');ax.set_xlabel('每人闭眼 − 睁眼的 α 占比差（百分点）');ax.set_ylabel('逻辑回归准确率（%）');ax.grid(alpha=.12)
    out['alpha']=image('alpha_difficulty',fig,'每人alpha占比差与基线准确率的描述性散点图')
    return out

METHOD_TEXT={
'lr':dict(no='01',title='逻辑回归：先建立简单参照',problem='建立可解释、计算快的基础分类器，判断频带能量本身能否区分状态。',benefit='只学习一组线性权重，复杂度低；便于判断后续更复杂模型的增益。',expect='获得明显高于多数类基线的跨人表现，作为固定对照。',steps=['20 维频带特征','训练折内标准化','L2 逻辑回归','睁眼概率 ≥ 0.5'],config='C=1；L2；solver=lbfgs；max_iter=2000；class_weight=balanced；random_state=42。',result='准确率 82.23%，比多数类基线 58.44% 高 23.79 个百分点。原基线逐窗预测完全复现。',cause='训练平均 83.45%，测试汇总 82.23%，间距较小。现有频域特征已有区分信息；部分受试者的分布偏移仍无法被统一边界处理。',limit='线性分类边界可能无法描述复杂特征组合。系数和相关特征不能直接解释为生理因果。',doc='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
'svm':dict(no='02',title='RBF SVM：检验非线性边界',problem='逻辑回归的直线式边界可能过于简单，频带之间的组合关系可能呈非线性。',benefit='RBF 核按特征距离衡量样本相似性，能够学习弯曲边界，适合检验非线性是否带来收益。',expect='希望超过逻辑回归，并减少原模型判错的边界样本；不预设提升幅度。',steps=['相同 20 维特征','训练折内标准化','RBF 核 SVM','决策分数 ≥ 0'],config='kernel=rbf；C=1；gamma=scale（由各训练折特征计算）；class_weight=balanced；probability=False；random_state=42。',result='准确率 82.72%，比基线高 0.49 个百分点，仅多判对 18 个窗口。睁眼召回率提高到 86.86%，闭眼召回率略降。',cause='非线性边界带来小幅变化，但困难受试者没有普遍改善。训练平均 86.48% 高于测试 82.72%，容量增加也可能拟合部分训练人群特征。',limit='输出使用决策分数计算 AUC，并未校准为概率。不能把分数当“睁眼概率”。与基线的配对差值区间跨零，未确认优势。',doc='https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'),
'rf':dict(no='03',title='随机森林：检验多特征组合',problem='单一边界可能忽略“某频带高、另一频带低”一类组合规则。',benefit='300 棵树分别学习条件规则，再平均各树的预测概率，减少单棵树的不稳定。',expect='希望利用特征交互、改善困难窗口，并在与 SVM 的对照中检验树模型是否更适合。',steps=['相同 20 维特征','相同标准化流程','300 棵随机决策树','平均概率 ≥ 0.5'],config='n_estimators=300；max_depth=None；min_samples_leaf=5；max_features=sqrt；bootstrap=True；class_weight=balanced；n_jobs=4；random_state=42。',result='准确率 81.77%，比基线低 0.46 个百分点。修复 114 个原错误，同时新增 131 个错误。',cause='训练平均 93.70%，明显高于测试 81.77%，与跨人泛化不足相符。复杂组合规则可能拟合了训练人群的局部模式；这是解释假设。',limit='增加树数不能自动解决人群差异。当前结果只代表这组固定参数，没有进行内层参数搜索。',doc='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
'hgb':dict(no='04',title='梯度提升树：逐步修正误差',problem='前三个模型都有较多共同错误，想检验分阶段修正误差能否找到额外区分规则。',benefit='后续树根据当前损失的梯度继续学习；较小学习率和受限叶节点控制每一步的复杂度。',expect='希望减少第一轮遗留的错误，尤其改善困难受试者；这一假设可能失败。',steps=['相同 20 维特征','训练折内标准化','200 轮梯度提升','睁眼概率 ≥ 0.5'],config='HistGradientBoostingClassifier；max_iter=200；learning_rate=0.05；max_leaf_nodes=15；min_samples_leaf=20；l2_regularization=1；class_weight=balanced；early_stopping=False；random_state=42。',result='准确率 82.12%，与基线几乎持平。修复 147 个原错误，但又新增 151 个错误，净少判对 4 窗。',cause='训练平均 94.08%，测试 82.12%，增加拟合能力未转化为跨人收益。部分困难者稍有改善，但其他窗口上的新增错误抵消了收益。',limit='关闭随机窗口验证早停，避免引入额外的被试内验证捷径；轮数固定。方案在看过第一轮结果后提出，属于探索性比较。',doc='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html'),
'stack':dict(no='05',title='Stacking：让模型学习如何融合',problem='第一轮模型错误有重叠，也有互补；想利用不同模型的分数，降低单个模型的判断偏差。',benefit='融合器从内部折外预测中学习如何结合 LR、SVM、随机森林，而不是手动选择对自己有利的权重。',expect='希望保留不同模型能修复的错误，同时减少新增错误；不保证超过最强单模型。',steps=['外层只取训练人群','内部按人 4 折产生 3 维分数','标准化 + LR 融合器','基模型重训后预测外层测试人'],config='基模型=LR / RBF SVM / 随机森林，参数与前实验一致；内部 GroupKFold=4；融合器 StandardScaler + LogisticRegression(C=1, L2, balanced, max_iter=2000)；外层沿用原 5 折；random_state=42。',result='准确率 82.64%，比基线高 0.41 个百分点，仍低于 SVM 0.08 个百分点。AUC 0.8913，为本次最高，但与 SVM 0.8910 非常接近。',cause='三者错误高度相关，融合器获得的新增信息有限。它修复 102 个基线错误、引入 87 个新错误，净多判对 15 窗；共同困难者依然突出。',limit='使用手写分组 Stacking，全部内部划分都有人员隔离记录。融合器只输入模型分数，不输入标签或人员编号。训练曲线仅作诊断；内部折外训练与最终基模型训练分数分布存在差异。',doc='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html')}

def method_tab(m):
    t=METHOD_TEXT[m];v=M[m]
    paired=next((x for x in A['paired_vs_lr'] if x['method']==m),None)
    cards=''.join(f'<article><span>{label}</span><h3>{title}</h3><p>{text}</p></article>' for label,title,text in [('问题','针对什么问题',t['problem']),('思路','为什么用它',t['benefit']),('假设','预期能达到什么',t['expect'])])
    steps='<div class="methodflow">'+''.join(f'<div><small>0{i+1}</small>{s}</div>' for i,s in enumerate(t['steps']))+'</div>'
    worst=sorted(v['subjects'],key=lambda s:s['accuracy'])[:3]
    rows=''.join(f'<tr><td>{s["subject"]}</td><td>{s["n"]}</td><td>{pc(s["accuracy"])}</td><td>{pc(s["recall_closed"])}</td><td>{pc(s["recall_open"])}</td><td>{DIAG[s["subject"]]["alpha_gap"]*100:+.2f} pp</td></tr>' for s in worst)
    pairtext=(f'相对 LR 的准确率差：<b>{pp(paired["delta_accuracy"])} 个百分点</b>；配对按人重采样 95% 区间 [{pp(paired["ci95"][0])}, {pp(paired["ci95"][1])}]。区间跨过零，本次不能确认提升。' if paired else '本页是后续所有模型的固定参照。重新训练的五折预测与原基线逐窗一致。')
    return heading('方法 '+t['no'],t['title'],'共用数据与特征保持不变。主结果来自训练时未见过的受试者。')+f'<div class="whygrid">{cards}</div>'+steps+f'<div class="config"><b>参数配置</b><p>{t["config"]}</p><small>数据清洗、两秒窗、20 维特征、外层五折均见“数据与控制”。标准化器只在对应训练折拟合。所有模型的类别权重从各自训练集计算。</small></div>'+statrow([('测试准确率',pc(v['accuracy'],2)),('平衡准确率',pc(v['balanced_accuracy'],2)),('宏平均 F1',f'{v["macro_f1"]:.3f}'),('ROC-AUC',f'{v["auc"]:.4f}')])+f'<div class="panel">{method_plot(m)}<p class="note">每个窗口只贡献一次外层测试预测；训练曲线用于发现过拟合，不作为模型最终成绩。</p></div><div class="callout">{pairtext}</div><div class="grid"><article class="panel"><h3>实验结果</h3><p>{t["result"]}</p><h3 style="margin-top:20px">可能的原因</h3><p>{t["cause"]}</p></article><article class="panel"><h3>方法的边界</h3><p>{t["limit"]}</p><p class="note">按人等权平均准确率 {pc(v["subject_accuracy_mean"])}；主指标按窗口加权。5 折拟合与预测合计 {v["training_seconds"]:.2f} 秒，为本机单次运行值，包含诊断预测；不等同于部署延迟。</p><a href="{t["doc"]}" target="_blank" rel="noopener">方法文档 ↗</a></article></div><div class="panel" style="margin-top:24px"><h3>本方法的三个低分受试者</h3><div class="tablewrap"><table><tr><th>受试者</th><th>窗口</th><th>准确率</th><th>闭眼召回</th><th>睁眼召回</th><th>α 差值</th></tr>{rows}</table></div><p class="note">α 差值 = 每人闭眼减睁眼的双通道平均相对功率，单位为百分点。它只是一条描述性线索。完整错误序列可在“诊断与改进”页切换模型查看。</p><button class="action" data-go="diagnosis">查看失败案例 →</button></div>'

def diagnostic_tab(ch):
    cards=''
    observations={'S0013':'闭眼 α 占比仅 13.70%，睁眼为 9.65%。基线将 53 个闭眼窗中的 48 个判成睁眼；相对群体，闭眼增强偏弱。',
      'S0043':'闭眼 α 占比 6.96%，睁眼 7.82%，方向与大多数人相反。基线闭眼召回仅 4.2%。这不能直接证明标签错误。',
      'S0037':'52 个闭眼窗被全部模型判为睁眼，46 个睁眼窗均判对。所有保留窗质量等级都是 0，但 α 占比差只有 1.93 个百分点。'}
    for s in observations:
        d=DIAG[s];cards+=f'<article><span>{s} · {d["n"]} 个窗口</span><h3>{d["all_models_wrong"]} 窗五模型共同错误</h3><p>{observations[s]}</p></article>'
    return heading('诊断 / 再设计','更复杂的边界，未解决共同难点','先看证据，再提出解释；以下分析不用于更改本次测试样本或标签。')+statrow([('第一轮三模型共同错','497 窗'),('全部五模型共同错',f'{A["all_wrong"]} 窗'),('共同错占全部窗口',pc(A['all_wrong']/3681)),('独立受试者','53 人')])+f'<div class="whygrid">{cards}</div><div class="grid" style="margin-top:24px"><div class="panel"><h3>低分人员在不同模型中重复出现</h3>{ch["heatmap"]}<p class="note">按原逻辑回归准确率从低到高选 12 人，未按新模型结果筛选。</p></div><div class="panel"><h3>α 差异较弱，是一个可见线索</h3>{ch["alpha"]}<p>群体平均闭眼 α 占比高于睁眼，但部分困难者的差异弱或反向。统一模型可能因此把他们的闭眼片段判成睁眼。</p><div class="callout warn">这是可能解释，不是因果证明。图使用评估数据做描述；不能据此擅自改标签或删除低分人员。</div><p class="note">原基线中，质量 0 窗准确率 81.8%，含质量 1 的窗 84.2%。当前统计不支持“所有错误都来自质量 1”这一简单解释；组成人员与类别不同，也不能解释为质量 1 更好。</p></div></div><div class="panel" style="margin-top:24px"><div class="inspecthead"><h3>错误序列检查器</h3><div><label>受试者 <select id="caseSubject">'+''.join(f'<option>{s}</option>' for s in ['S0013','S0043','S0037','S0027','S0030'])+'</select></label> <label>模型 <select id="caseModel">'+''.join(f'<option value="{m}">{NAMES[m]}</option>' for m in ORDER)+'</select></label></div></div><p id="caseStats"></p><div id="caseTimeline" class="timeline"></div><p class="note">每格 = 保留的一个两秒窗口，按文件位置排序；格内“闭/睁”为真实标签。红色代表判错，蓝绿代表判对。鼠标停留可看预测、session 和行号。被删除的数据没有补入，因此横向位置不是连续时间轴。</p></div><h2 style="margin-top:36px">由第一轮结果选择两条改进路径</h2><div class="grid"><div class="panel"><h3>路径一：梯度提升</h3><p>前三个模型共同错误多，检验逐步修正损失能否补充有效规则。固定浅树容量与学习率，不根据外层测试分数反复调参。</p><button class="action" data-go="hgb">查看梯度提升实验 →</button></div><div class="panel"><h3>路径二：分组 Stacking</h3><p>SVM 能修复部分 LR 错误，也会引入新错误；检验模型分数融合能否利用互补性。内层 4 折也按人划分，禁止测试人进入融合训练。</p><button class="action" data-go="stack">查看 Stacking 实验 →</button></div></div><div class="callout warn">两种方案是在看过第一轮结果后提出，继续使用相同外层五折，属于探索性比较。内层人员隔离可避免训练泄漏，但不能消除方案选择带来的乐观偏差；最终改善仍需新受试者确认。</div><details><summary>为什么这次没有直接上深度学习</summary><p class="note">本次有 3,681 个窗口，但独立受试者只有 53 人。先用相同特征比较集成方法，能更清楚地分辨“分类器变化”的作用。原始波形 CNN 会同时改变特征学习和分类器，还需要按人验证的早停与正则化设计；可作为后续单列实验，并不意味着深度学习一定无效。</p></details>'

def comparison_tab(ch):
    rows=''
    for m in ORDER:
        v=M[m];pair=next((p for p in A['paired_vs_lr'] if p['method']==m),None)
        rows+=f'<tr><td><b>{NAMES[m]}</b></td><td>{pc(v["accuracy"],2)}</td><td>{pc(v["balanced_accuracy"],2)}</td><td>{v["macro_f1"]:.4f}</td><td>{v["auc"]:.4f}</td><td>{pc(v["subject_accuracy_mean"],2)}</td><td>{pp(pair["delta_accuracy"]) if pair else "参照"}</td></tr>'
    errorrows=''.join(f'<tr><td>{NAMES[p["method"]]}</td><td>+{p["fixed_baseline_errors"]}</td><td>−{p["new_errors"]}</td><td>{p["fixed_baseline_errors"]-p["new_errors"]:+}</td></tr>' for p in A['paired_vs_lr'])
    fullrows=''.join('<tr><td>'+d['subject']+'</td><td>'+str(d['n'])+'</td>'+''.join(f'<td>{pc(d["accuracy"][m])}</td>' for m in ORDER)+'</tr>' for d in sorted(A['diagnostics'],key=lambda d:d['accuracy']['lr']))
    return heading('最终对照 / 结论','五个模型集中在 82% 左右','SVM 的准确率略高，Stacking 的 AUC 略高；当前证据不足以确认复杂模型有稳定优势。')+f'<div class="panel"><div class="tablewrap"><table class="comparison-table"><tr><th>模型</th><th>准确率</th><th>平衡准确率</th><th>宏 F1</th><th>AUC</th><th>按人平均准确率</th><th>较 LR / pp</th></tr>{rows}</table></div><p class="note">全部使用同一批 53 人 / 3,681 窗 / 20 维特征 / 外层 5 折。正类为睁眼。多数类基线：准确率 58.44%，平衡准确率 50.00%。</p></div><div class="grid" style="margin-top:24px"><div class="panel"><h3>总体区间高度重叠</h3>{ch["accuracy"]}</div><div class="panel"><h3>配对差值区间均跨过零</h3>{ch["delta"]}</div></div><p class="note">区间用固定折外预测按人有放回重采样 3,000 次（seed=42），不是按窗口重采样；差值使用同一批抽中的人作配对计算。未在每次重采样中重训模型，不包括训练随机性或方法选择不确定性。区间跨零不等于模型完全等效。</p><div class="grid"><div class="panel"><h3>新增能力被新错误抵消</h3><table><tr><th>较逻辑回归</th><th>修复原错误</th><th>引入新错误</th><th>净变化</th></tr>{errorrows}</table><p class="note">最高准确率仅比基线多判对 18 个窗口。五个模型仍共同判错 461 窗，说明优先复核共同困难者比继续堆模型更有信息价值。</p></div><div class="panel"><h3>当前保留哪个方案</h3><p><b>保留逻辑回归作主基线，SVM 作候选方案。</b>前者简单、可复现；后者小幅提升睁眼召回率，仍需独立验证。</p><p>Stacking 的额外训练过程未带来清晰收益。随机森林和梯度提升训练分数明显较高，但测试分数接近基线，暂没有替换主模型的依据。</p><div class="callout">方法复杂度不是结果保证。此次改进最明确的产出，是定位共同失败模式与下一轮验证问题。</div></div></div><h2 style="margin-top:36px">下一轮优化：每次只改一个因素</h2><div class="roadmap"><article><b>01</b><h3>核查共同 badcase</h3><p>检查困难者的原始波形、状态切换、导联与残余眼电。如有异常，先记录独立质量依据，不能因为预测错误而删数据。</p><small>验证：盲于预测结果的质量复核。</small></article><article><b>02</b><h3>检验更稳健的特征</h3><p>固定模型与人员划分，比较仅相对功率、α/θ 比值、通道一致性或更细的 α 峰特征。每次明确是哪一项改变。</p><small>验证：完整内层按人选择，外层评价。</small></article><article><b>03</b><h3>分清跨人与个体校准</h3><p>可尝试少量个人校准或无标签归一化，但这是新的应用条件。不能用测试人的全部有标签数据估计归一化或阈值。</p><small>验证：显式划分校准段与后续测试段。</small></article><article><b>04</b><h3>增加独立受试者验证</h3><p>冻结最终流程后，在新受试者上一次性评价。若尝试 CNN，单独报告波形输入、正则化、按人早停与随机种子波动。</p><small>验证：外部留出集与多种子重复。</small></article></div><details><summary>展开全部 53 人的模型对照表</summary><div class="tablewrap"><table><tr><th>受试者</th><th>窗口</th>'+''.join(f'<th>{n}</th>' for n in NAMES.values())+f'</tr>{fullrows}</table></div></details><details><summary>复现、实验边界与来源</summary><div class="source-grid"><div><b>运行顺序</b><br><code>python run_baseline.py</code><br><code>python compare_models.py --stage 1</code><br><code>python compare_models.py --stage 2</code><br><code>python build_comparison.py</code><p>在脚本目录运行即可；脚本按自身路径定位项目数据。原始清洗后数据、逐窗预测、嵌套人员划分、参数和输入哈希保存在本地 comparison 与 results 目录。</p></div><div><b>来源与限制</b><p>课程 V5 数据说明第 12–19 页；EEG 常用特征课件第 33、35–36 页；训练实验范式第 10–11 页。具体模型方法文档见各页签。逻辑回归概率、树模型概率及 SVM 分数均未另做概率校准。AUC 汇总跨折得分，非单一最终模型的独立测试分数。</p><p>结果仅适用于筛选后的较干净片段，未验证在线、其他设备或新的数据集。量表不参与模型；所有因果解释均仍待验证。</p></div></div></details>'

def main():
    current=B/'睁闭眼识别_实验汇报.html';archive=B/'睁闭眼识别_基线存档.html'
    if not archive.exists():shutil.copy2(current,archive)
    old=archive.read_text(encoding='utf-8');style=re.search(r'<style>(.*?)</style>',old,re.S).group(1)
    ch=overview_plots()
    data=re.search(r'<section id="data">.*?</section>',old,re.S).group(0)
    features=re.search(r'<section id="features">.*?</section>',old,re.S).group(0)
    control=heading('课程项目 / 实验扩展 · 2026.11.17','睁闭眼识别 · 五种方法对照','沿用原基线，先比较两种常见模型，再根据失败模式尝试两种集成方法。')+statrow([('参与对照','53 人'),('相同两秒窗口','3,681'),('相同输入特征','20 维'),('按人隔离评价','5 折')])+'''<div class="callout"><b>控制变量：</b>人员、窗口、标签、清洗门槛、频域特征及外层五折名单全部固定；改变分类方法及其固定参数。原基线预测已逐窗复现。</div><div class="grid"><div class="panel"><h3>共同数据处理</h3><p>课程已做带通、陷波和去眼电。本次按 readme 排除坏数据；状态段首尾各去 1 秒；2 秒无重叠窗；剔除含质量 2/3、|幅值| &gt;100 μV 或标准差 &lt;0.1 μV 的窗口。</p><p>去均值 → Welch（1 秒 Hann 子窗，50% 子窗重叠，FFT=250）→ 两通道 × 五频带 × log绝对功率/相对功率 = 20 维。</p></div><div class="panel"><h3>共同训练与评估</h3><p>沿用原 GroupKFold 人员名单，每折训练 42–43 人、测试 10–11 人。所有模型的标准化只拟合训练折；树模型也沿用同一流程。</p><p>使用类别平衡权重，固定随机种子 42，不做参数搜索，不因模型预测改变样本或标签。Stacking 额外使用训练人群内部 4 折。</p></div></div><div class="sequence"><button data-go="lr">01 基线 LR</button><span>→</span><button data-go="svm">02 SVM</button><button data-go="rf">03 随机森林</button><span>→</span><button data-go="diagnosis">失败分析</button><span>→</span><button data-go="hgb">04 梯度提升</button><button data-go="stack">05 Stacking</button></div>'''+data+features
    content={'data':control,'lr':method_tab('lr'),'svm':method_tab('svm'),'rf':method_tab('rf'),'diagnosis':diagnostic_tab(ch),'hgb':method_tab('hgb'),'stack':method_tab('stack'),'compare':comparison_tab(ch)}
    labels={'data':'数据与控制','lr':'01 逻辑回归','svm':'02 SVM','rf':'03 随机森林','diagnosis':'诊断与改进','hgb':'04 梯度提升','stack':'05 Stacking','compare':'全部对比'}
    if (B/'tuning/metrics.json').exists():
        from tuning_report import enhance
        content,labels=enhance(content,labels)
    css='''
[hidden]{display:none!important}header{position:sticky}.shell{max-width:1240px;margin:auto;padding:0 28px}.topline{display:flex;align-items:center;justify-content:space-between;padding:15px 0 10px}.topline b{font-size:14px}.topline span{font-size:12px;color:var(--muted)}.tabs{display:flex;gap:3px;overflow-x:auto;padding-bottom:10px;scrollbar-width:thin}.tabs button{border:0;border-radius:5px;background:transparent;color:#637985;font-size:13px;padding:10px 14px;white-space:nowrap}.tabs button[aria-selected=true]{background:#182d3d;color:white}.tabs button:focus-visible{outline:3px solid #51afa6}main{max-width:1240px}.tabpane{padding-bottom:25px}.intro{padding:43px 0 24px}.intro h1{font-size:36px;letter-spacing:-1px}.intro .lead{max-width:none}.tabpane>h2{font-size:25px}.whygrid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.whygrid article{padding:22px;background:white;border:1px solid var(--line);border-radius:10px}.whygrid span{font-size:11px;color:var(--teal);font-weight:700}.whygrid h3{margin:8px 0}.whygrid p,.panel>p{font-size:14px;color:#506773;line-height:1.9}.methodflow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}.methodflow div{border-top:3px solid var(--teal);background:#e7f0ef;padding:15px;font-size:13px;font-weight:600}.methodflow small{display:block;color:var(--teal);margin-bottom:6px}.config{padding:19px 23px;margin:0 0 24px;border:1px dashed #afc4c8;border-radius:8px}.config p{font-size:13px;margin:4px 0;line-height:1.9}.config small{font-size:12px;color:var(--muted)}.resultmetrics strong{font-size:30px}.sequence{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:28px 0 8px}.sequence button{background:#e1edeb;border:0;padding:8px 14px;border-radius:6px;color:#176d67;font-size:12px}.sequence span{color:#a1b2b6}.tabpane .tablewrap{max-height:520px}.tabpane .chart{margin-top:12px}.tabpane>section{padding-top:30px}.inspecthead{display:flex;align-items:center;justify-content:space-between;gap:14px}.inspecthead label{font-size:12px}.inspecthead select{padding:5px;border:1px solid #c8d4d6;border-radius:5px;background:white;color:var(--ink)}.timeline{display:flex;gap:5px;flex-wrap:wrap;margin:15px 0}.epoch{width:29px;height:28px;font-size:12px;line-height:28px;text-align:center;border-radius:4px;background:#d8eeea;color:#13786e}.epoch.bad{background:#f5d8cf;color:#a44530}.epoch.break{margin-left:10px;border-left:3px solid #182d3d}.roadmap{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.roadmap article{background:white;border-top:3px solid #182d3d;padding:20px}.roadmap b{font-size:12px;color:var(--teal)}.roadmap h3{font-size:16px;margin:9px 0}.roadmap p{font-size:13px;color:#526873}.roadmap small{font-size:11px;color:var(--teal)}.comparison-table td:nth-child(2){font-weight:700;color:var(--teal)}.tablewrap{overflow-x:auto}footer{align-items:center}footer a{font-size:12px}.chart{background:white}#caseStats{font-size:14px}body .callout{font-size:14px}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
@media(max-width:900px){.whygrid{gap:12px}.whygrid article{padding:16px}.roadmap{grid-template-columns:repeat(2,1fr)}.tabs button{padding:9px 11px}.intro h1{font-size:30px}.inspecthead{display:block}.inspecthead h3{margin-bottom:12px}}
@media(max-width:600px){.shell{padding:0 18px}.topline span{display:none}.intro{padding:25px 0 15px}.intro h1{font-size:28px}.whygrid{grid-template-columns:1fr}.methodflow{grid-template-columns:1fr 1fr}.resultmetrics strong{font-size:25px}.resultmetrics div{padding:14px}.roadmap{grid-template-columns:1fr}.config{padding:16px}.tablewrap table{min-width:570px}.tabpane>.grid{gap:15px}.panel>p{font-size:13px}}
@media print{.shell,.sequence,.inspecthead select,button{display:none}.tabpane[hidden]{display:none!important}.intro{padding-top:0}.intro h1{font-size:28px}.whygrid article,.roadmap article{break-inside:avoid}.tablewrap{max-height:none!important;overflow:visible}.chart{max-height:none}.tabpane>section{break-inside:auto}}
'''
    case_data={s:{m:PREDS[m][PREDS[m].subject==s][['session','start_row','label','prediction','correct']].to_dict('records') for m in ORDER} for s in ['S0013','S0043','S0037','S0027','S0030']}
    js='''
const tabs=[...document.querySelectorAll('[role=tab]')];
function showTab(id,scroll=true){if(!document.getElementById('panel-'+id))id='data';tabs.forEach(b=>{let on=b.dataset.tab===id;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;document.getElementById('panel-'+b.dataset.tab).hidden=!on;});history.replaceState(null,'','#'+id);if(scroll)window.scrollTo(0,0);document.title='睁闭眼识别 · '+document.querySelector('[data-tab="'+id+'"]').textContent;}
tabs.forEach((b,i)=>{b.addEventListener('click',()=>showTab(b.dataset.tab));b.addEventListener('keydown',e=>{let k;if(e.key==='ArrowRight')k=(i+1)%tabs.length;else if(e.key==='ArrowLeft')k=(i+tabs.length-1)%tabs.length;else if(e.key==='Home')k=0;else if(e.key==='End')k=tabs.length-1;if(k!==undefined){e.preventDefault();showTab(tabs[k].dataset.tab);tabs[k].focus();}})});
document.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.go)));
function showCase(){let s=document.getElementById('caseSubject').value,m=document.getElementById('caseModel').value,r=CASES[s][m];let hits=r.reduce((a,x)=>a+x.correct,0);document.getElementById('caseStats').textContent=s+'：'+hits+' / '+r.length+' 窗判对，准确率 '+(100*hits/r.length).toFixed(1)+'%。';let box=document.getElementById('caseTimeline');box.replaceChildren();r.forEach((v,i)=>{let e=document.createElement('span');e.className='epoch'+(v.correct?'':' bad')+(i>0&&r[i-1].session!==v.session?' break':'');e.textContent=v.label?'睁':'闭';e.title=v.session+' · 第 '+v.start_row+' 行起（零起点） · 真实'+(v.label?'睁眼':'闭眼')+' · 预测'+(v.prediction?'睁眼':'闭眼');box.append(e);});}
document.getElementById('caseSubject').addEventListener('change',showCase);document.getElementById('caseModel').addEventListener('change',showCase);showCase();showTab(location.hash.slice(1),false);window.addEventListener('hashchange',()=>showTab(location.hash.slice(1)));
'''
    result='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="脑机接口课程项目：五种机器学习方法的睁闭眼识别对照、失败案例与优化思路。"><title>睁闭眼识别 · 五模型对照</title><style>'+style+css+'</style></head><body><header><div class="shell"><div class="topline"><b>BCI / 睁闭眼识别实验</b><span>相同数据 · 相同人员划分 · 五种方法</span><button class="action" onclick="window.print()">打印当前页签</button></div><div class="tabs" role="tablist" aria-label="报告页签">'+''.join(f'<button id="tab-{k}" role="tab" data-tab="{k}" aria-controls="panel-{k}" aria-selected="{str(i==0).lower()}" tabindex="{0 if i==0 else -1}">{v}</button>' for i,(k,v) in enumerate(labels.items()))+'</div></div></header><main>'+''.join(f'<div class="tabpane" id="panel-{k}" role="tabpanel" aria-labelledby="tab-{k}" {"hidden" if i else ""}>{v}</div>' for i,(k,v) in enumerate(content.items()))+'<footer><span>脑机接口课程项目 · 2026.11.17 · 所有图表已内嵌</span><a href="https://github.com/Mr-yangtuo/Mr-yangtuo.github.io/tree/main/eeg-eye-state" target="_blank" rel="noopener">报告与复现代码 ↗</a></footer></main><script>const CASES='+json.dumps(case_data,ensure_ascii=False)+';'+js+'</script></body></html>'
    if (B/'tuning/metrics.json').exists():
        result=result.replace('相同数据 · 相同人员划分 · 五种方法','相同数据 · 嵌套调参 · 逐项对照').replace('睁闭眼识别 · 五模型对照','睁闭眼识别 · 嵌套调参对照')
    current.write_text(result,encoding='utf-8');(O/'index.html').write_text(result,encoding='utf-8')
    print('Saved',current,'bytes',len(result.encode('utf-8')))

if __name__=='__main__':main()
