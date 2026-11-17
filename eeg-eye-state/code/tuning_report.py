"""Report extension for the completed nested search; historical reports remain visible."""
from pathlib import Path
import json,io,base64,html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;O=B/'tuning'
N={'lr':'逻辑回归','svm':'RBF SVM','rf':'随机森林','hgb':'梯度提升树','svm_select':'SVM + 特征选择'}
C=['#078b89','#346cac','#b78538','#9c5a7d','#615bb8']
def pct(v):return f'{v*100:.2f}%'
def table(headers,rows):return '<div class="tablewrap"><table><tr>'+''.join('<th>'+s+'</th>' for s in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</table></div>'
def pic(name,fig,alt):
    out=O/'figures';out.mkdir(exist_ok=True);fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');buf=io.BytesIO();fig.savefig(buf,format='svg',bbox_inches='tight');plt.close(fig)
    return '<img class="chart" alt="'+alt+'" src="data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
def head(title,sub):return '<div class="intro"><div class="eyebrow">强化调参 / 嵌套按人验证</div><h1>'+title+'</h1><p class="lead">'+sub+'</p></div>'
def stats(v):return '<div class="resultmetrics">'+''.join(f'<div><strong>{x}</strong><span>{n}</span></div>' for n,x in [('测试准确率',pct(v['accuracy'])),('平衡准确率',pct(v['balanced_accuracy'])),('每人等权平衡准确率',pct(v['subject_balanced_accuracy'])),('ROC-AUC',f'{v["auc"]:.4f}')])+'</div>'
def method_chart(key,v,old):
    fig,(a,b)=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained',gridspec_kw={'width_ratios':[.9,1.3]})
    cm=np.array(v['cm']);a.imshow(cm/cm.sum(axis=1,keepdims=True),cmap='BuGn',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):a.text(j,i,f'{cm[i,j]}\n{cm[i,j]/cm[i].sum()*100:.1f}%',ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.6 else '#182d3d',fontsize=13)
    a.set_xticks([0,1],['预测闭眼','预测睁眼']);a.set_yticks([0,1],['真实闭眼','真实睁眼']);a.set_title('调参 + 阈值：外层测试混淆矩阵',fontsize=12)
    base=old['svm' if key=='svm_select' else key]
    b.plot(range(1,6),[q['accuracy']*100 for q in base['folds']],'o--',color='#a2afb5',label='原固定参数')
    b.plot(range(1,6),[q['metrics']['accuracy']*100 for q in v['folds']],'o-',color='#078b89',label='嵌套调参 + 阈值')
    b.set_xticks(range(1,6),[f'第 {i} 折' for i in range(1,6)]);b.set_ylim(65,100);b.set_ylabel('准确率（%）');b.legend(frameon=False,fontsize=10);b.grid(alpha=.15)
    return pic(key+'_tuned',fig,N[key]+'调参混淆矩阵与逐折原参数对照')
def compare_chart(m,old):
    fig,ax=plt.subplots(figsize=(10.8,3.6),layout='constrained');x=np.arange(5);w=.24
    vs=[[old['svm' if k=='svm_select' else k]['accuracy']*100 for k in N], [m[k]['default_threshold_metrics']['accuracy']*100 for k in N],[m[k]['accuracy']*100 for k in N]]
    for off,vals,color,label in zip([-w,0,w],vs,['#c2cdd1','#759db8','#078b89'],['原固定参数','仅调参数','再调阈值']):
        bars=ax.bar(x+off,vals,w,color=color,label=label)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+w/2,val+.25,f'{val:.2f}',ha='center',va='bottom',fontsize=9,rotation=60)
    ax.set_xticks(x,list(N.values()));ax.set_ylabel('外层准确率（%）');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.03))
    return pic('tuning_ablation',fig,'固定参数、嵌套参数搜索、附加阈值调整的准确率比较')
def delta_chart(m):
    fig,ax=plt.subplots(figsize=(6,3.8),layout='constrained')
    for i,k in enumerate(N):
        q=m[k]['comparisons']['svm'];x=q['delta_accuracy']*100;lo,hi=np.array(q['ci95'])*100
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=C[i],capsize=4);ax.annotate(f'{x:+.2f}',(x,i),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10)
    ax.axvline(0,color='#8a999f',ls='--');ax.set_yticks(range(5),list(N.values()));ax.invert_yaxis();ax.set_xlabel('相对上一版 SVM 的准确率差（百分点）');ax.grid(axis='x',alpha=.12)
    return pic('tuning_delta',fig,'与原SVM配对按人重采样的准确率差区间')
def parameter_rows(v):
    rows=[]
    modes={'balanced':'类别平衡','none':'无额外权重','subject_balanced':'每人×每类等权'}
    fm={'all':'全部20维','relative':'相对功率10维','logpower':'log绝对功率10维'}
    for f in v['folds']:
        p=f['best_params'];ps='；'.join(k+'='+str(round(x,6) if isinstance(x,float) else x) for k,x in p.items() if k not in ['features','weight_mode'])
        rows.append([f['fold'],fm[p['features']],modes[p['weight_mode']],ps,f'{f["threshold"]:.3f}',pct(f['best_inner_subject_ba']),pct(f['metrics']['accuracy'])])
    return table(['外折','特征','训练权重','所选参数','阈值','内层选参分数','外层准确率'],rows)
def tuned_method(k,v,old):
    base='svm' if k=='svm_select' else k;q=v['comparisons'][base]
    direction='提高' if q['delta_accuracy']>=0 else '下降'
    ci=q['ci95'];certainty='差值区间跨零，尚不能确认稳定提升。' if ci[0]<=0<=ci[1] else ('差值区间在零以上，但重复使用本批受试者设计方案仍需独立验证。' if ci[0]>0 else '这组搜索方案整体变差，不能因为内层分数较高就采用。')
    worst=sorted(v['subjects'],key=lambda x:x['accuracy'])[:3]
    cause={'lr':'C 改变正则化强度，训练权重改变不同人/类别对损失的贡献。即便内层分数提高，测试人员的信号分布也可能与内层人员不同。',
      'svm':'C 与 γ 共同控制边界复杂度。全局搜索加局部细化可找到原默认值之外的方案；外层表现取决于这种边界能否迁移到新的人。',
      'rf':'树深、叶节点大小和抽样比例限制局部规则的复杂度。若训练与测试仍有大间距，说明单靠正则化尚不足以消除人群差异。',
      'hgb':'学习率、轮数、叶子数量和 L2 同时影响容量。搜索可以抑制过拟合，但可能同时削弱可区分信号，因此必须看外层结果。',
      'svm_select':'只用相对功率可能减少幅值尺度的个体差异，也可能丢掉有用的绝对能量信息；内层选择哪种特征，并不保证对外层受试者最好。'}[k]
    if k=='svm_select':
        selected=[f['best_params']['features'] for f in v['folds']]
        cause+=f' 本次实际选择：{selected.count("logpower")} 折仅 log 绝对功率、{selected.count("all")} 折全部特征、{selected.count("relative")} 折仅相对功率。因此，当前结果不支持“只用相对功率就能明显解决个体差异”的预期；减少特征冗余可能有帮助，但尚未确认。'
    if k in ['rf','hgb']:
        cause+=f' 本轮训练平均准确率从原来的 {pct(old[k]["train_accuracy_mean"])} 降至 {pct(v["train_accuracy_mean"])}，说明搜索确实选出了更受约束的模型，但测试表现没有同步提高。'
    return head(N[k]+'：嵌套调参结果','每折单独选择参数和阈值；下方公布全部五折的选择，不挑最好的一折。')+stats(v)+f'<div class="panel">{method_chart(k,v,old)}</div><div class="callout">相对原{N[base]}准确率{direction} <b>{abs(q["delta_accuracy"])*100:.2f} 个百分点</b>；配对按人重采样区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]。{certainty}</div><div class="grid"><div class="panel"><h3>参数与阈值分别贡献了什么</h3>'+table(['实验','准确率','平衡准确率'],[['原固定参数',pct(old[base]['accuracy']),pct(old[base]['balanced_accuracy'])],['选参后 / 默认阈值',pct(v['default_threshold_metrics']['accuracy']),pct(v['default_threshold_metrics']['balanced_accuracy'])],['选参后 / 内层调阈值',pct(v['accuracy']),pct(v['balanced_accuracy'])]])+f'<p class="note">阈值没有改变排序，所以同一参数模型的 AUC 不变。阈值目标也是每人等权平衡准确率，不保证普通准确率增加。</p></div><div class="panel"><h3>实验分析</h3><p>{cause}</p><p>修复原模型 {q["fixed"]} 个错误，同时新增 {q["new_errors"]} 个错误。训练平均准确率 {pct(v["train_accuracy_mean"])}，外层汇总准确率 {pct(v["accuracy"])}。</p><p class="note">因果解释仍是待验证假设；没有因预测错误而删除人员、改变标签或放松质量门槛。</p></div></div><div class="panel" style="margin-top:24px"><h3>五折最终参数与阈值</h3>{parameter_rows(v)}<p class="note">内层选参分数 = 各受试者的两类召回率均值再按人平均；和外层窗口准确率并非同一指标，不能直接相减解释为过拟合。</p></div><div class="panel" style="margin-top:24px"><h3>调参后仍困难的三个人</h3>'+table(['受试者','窗口','准确率','闭眼召回','睁眼召回'],[[s['subject'],s['n'],pct(s['accuracy']),pct(s['recall_closed']),pct(s['recall_open'])] for s in worst])+'</div>'

def enhance(content,labels):
    m=json.loads((O/'metrics.json').read_text());old=json.loads((B/'comparison/metrics.json').read_text());p=json.loads((O/'protocol.json').read_text())
    total=sum(v['inner_candidate_evaluations'] for v in m.values());fits=(total-5)*4+25
    best=max(m,key=lambda k:m[k]['accuracy']);defaultbest=max(m,key=lambda k:m[k]['default_threshold_metrics']['accuracy'])
    ranges=[['逻辑回归','C：10⁻⁴–10³，15 档 × 3 种权重','45 组 / 外折'],['RBF SVM','C：10⁻³–10³；γ：10⁻⁴–10；64 组对数随机采样 + 3 组锚点；前两名附近按 ×0.3/1/3 细化','约 83 组 / 外折'],['随机森林','深度 3/5/8/12/不限；最小叶 1/3/5/10/20/40；特征抽样 sqrt/0.5/1；样本抽样 0.6/0.8/1；300 棵树','49 组 / 外折'],['梯度提升','轮数 100/200/400；学习率 .025/.05/.1；叶子 3/7/15/31；最小叶 10/20/40/80；L2 0/1/10/100；关闭早停','49 组 / 外折'],['SVM + 特征选择','相对功率10维、log绝对功率10维、全部20维；前两种各重复 SVM 搜索，全部特征复用同折最佳结果','约 151 组 / 外折']]
    strategy=head('从固定参数，升级为嵌套搜索','本轮重新运行四种基础模型，另外加入 SVM 特征子集选择；原分组 Stacking 保留作第一轮参照。')+ '<div class="resultmetrics">'+''.join(f'<div><strong>{v}</strong><span>{k}</span></div>' for k,v in [('外层 / 内层','5 × 4'),('候选配置评估',f'{total:,}'),('模型拟合次数',f'{fits:,}'),('数据仍保持','53 人 / 3,681 窗')])+'</div><div class="whygrid"><article><span>问题</span><h3>固定参数可能错过合适容量</h3><p>第一轮没有搜索 C、γ 和树的复杂度；更复杂模型的训练分数较高，测试提升很小。需要在训练人群中系统选择正则化。</p></article><article><span>预期</span><h3>改善跨人识别，不预设提升</h3><p>希望获得较好的泛化边界，并检验原参数是否限制了表现。搜索更充分也可能选出对新人员无效的参数，最终结果不保证提高。</p></article><article><span>验证</span><h3>用外层测试搜索策略本身</h3><p>每个外层测试组只用于最后评分。内层选参数、训练权重与阈值；所有搜索完成后才统一计算外层指标。</p></article></div><div class="methodflow"><div><small>01</small>固定原五折人员名单</div><div><small>02</small>外层训练人群内再分四折</div><div><small>03</small>广域搜索 → SVM 局部细化 → 阈值选择</div><div><small>04</small>用该外层全部训练人重训，预测未见人员</div></div><div class="panel"><h3>搜索范围与预算</h3>'+table(['模型','范围','预算'],ranges)+'</div><div class="grid" style="margin-top:24px"><div class="panel"><h3>每个人、每种状态等权</h3><p>内层目标：先求每个人的平衡准确率（闭眼召回与睁眼召回平均），再对人取平均。这样不会让窗口多的人主导选择。</p><p>训练权重尝试：不加权、类别平衡、受试者×类别平衡。第三种按训练集中每人每类的样本数取倒数，并归一化到平均权重 1。</p><p class="note">公平目标与旧报告窗口准确率不同，所有指标都公开展示。</p></div><div class="panel"><h3>阈值在内层单独选择</h3><p>先按默认阈值选最佳参数，再用所选模型的内层折外分数调阈值。SVM 搜索 −1 到 1（步长 .05），概率模型 .2 到 .8（步长 .02）。并列时优先接近默认阈值。</p><p>标准化只在各内层训练折拟合。SVM 特征选择另列为附加实验；其他四种保持原 20 维输入。</p></div></div><div class="callout warn">本轮搜索空间参考了前两轮结果，因此不是从未见过数据的确认性实验。内外层隔离能防止此次拟合泄漏，但不能消除反复分析同一批人带来的方案选择偏差；最终提升仍需独立受试者验证。</div><p class="note">随机种子 42；RF 每次 300 棵树；八进程搜索，每个进程限制一个计算线程。候选数含五次全特征 SVM 最佳结果复用，拟合次数已扣除这五次复用。<a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">嵌套验证文档</a> · <a href="https://scikit-learn.org/stable/modules/classification_threshold.html">阈值选择文档</a></p>'
    content['data']='<div class="callout">本页保留原数据清洗与特征说明。下文“固定参数、不做搜索”描述第一轮；本轮搜索见“强化调参”。样本与外层人员划分保持不变。</div>'+content['data']
    for k in ['lr','svm','rf','hgb']:
        history=content[k];content[k]=tuned_method(k,m[k],old)+'<details><summary>第一轮固定参数实验（思路与原结果存档）</summary>'+history+'</details>'
    content['stack']='<div class="callout">此页为第一轮分组 Stacking 参照。本轮集中调优四个基础模型及 SVM 特征子集，未重新搜索 Stacking 的多层融合过程。</div>'+content['stack']
    oldcompare=content['compare']
    rows=[]
    for k in N:
        ref=old['svm' if k=='svm_select' else k];v=m[k]
        rows.append([N[k],pct(ref['accuracy']),pct(v['default_threshold_metrics']['accuracy']),pct(v['accuracy']),pct(v['balanced_accuracy']),pct(v['subject_balanced_accuracy']),f'{v["auc"]:.4f}'])
    q=m[best]['comparisons']['svm'];ci=q['ci95']
    verdict='相对原 SVM 的配对差值区间跨过零，尚不能确认稳定提升。' if ci[0]<=0<=ci[1] else ('本次差值区间为正，但仍需独立新人员验证。' if ci[0]>0 else '本次搜索的最佳结果仍低于旧 SVM，不建议替换。')
    groups=[]
    for s in sorted(m['lr']['subjects'],key=lambda x:x['accuracy'])[:12]:
        sid=s['subject'];vals=[next(x for x in m[k]['subjects'] if x['subject']==sid)['accuracy'] for k in N];groups.append([sid]+[pct(v) for v in vals])
    content['compare']=head('调参前后，逐项拆开比较','相同人员、相同窗口。四种纯参数搜索与 SVM 特征选择分开列示；两种阈值版本都保留，避免只报告较好的版本。')+f'<div class="panel">{table(["模型","原固定参数","仅调参数","再调阈值","平衡准确率¹","每人等权平衡准确率¹","AUC"],rows)}<p class="note">¹ 两列为“再调阈值”版本。特征选择行以原 SVM 作参照；其输入维数可在内层选为10或20维，其余保持20维。原 Stacking：准确率82.64%、AUC0.8913。</p></div><div class="panel" style="margin-top:24px">{compare_chart(m,old)}</div><div class="grid" style="margin-top:24px"><div class="panel"><h3>相对上一版最好的单模型</h3>{delta_chart(m)}<p class="note">参照为原 SVM 82.72%；以人为单位配对重采样3000次，使用固定折外预测。区间不包含训练随机性与方案选择不确定性。</p></div><div class="panel"><h3>如何解读这次结果</h3><p>调参+阈值版本中，准确率最高为 <b>{N[best]} {pct(m[best]["accuracy"])}</b>；默认阈值版本最高为 <b>{N[defaultbest]} {pct(m[defaultbest]["default_threshold_metrics"]["accuracy"])}</b>。这两个“最高”只是外层结果的描述，不是新的未偏评价。</p><p>最佳调阈值方案相对原 SVM 变化 {q["delta_accuracy"]*100:+.2f} 个百分点，95%差值区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]。{verdict}</p><p>内层最优不必然是外层最优；阈值也可能用普通准确率换取更均衡的人员/类别表现。因此不能把更多搜索直接等同于模型更有效。</p></div></div><div class="panel" style="margin-top:24px"><h3>新一轮低分人员对照</h3>'+table(['受试者']+list(N.values()),groups)+'</div><div class="callout">下一步优先核查共同失败的信号与标签边界，再做更精细的频谱/时域特征消融。若尝试个人校准或时间平滑，需要单独定义使用条件；不能利用真实状态标签来平滑测试预测。</div><details><summary>完整第一轮五模型对照（存档）</summary>'+oldcompare+'</details><details><summary>复现本轮与核验</summary><p><code>python tune_models.py --workers 8</code> → <code>python verify_tuning.py</code> → <code>python build_comparison.py</code>。tuning 目录保存每折每模型的所有候选分数、选定阈值、人员名单、内层折外分数、外层预测和模型文件；训练可以从已完成的折恢复。参数选择与阈值只使用该外折训练人群。</p></details>'
    content['svm_select']=tuned_method('svm_select',m['svm_select'],old)
    content['tuning']=strategy
    order=['data','tuning','lr','svm','rf','hgb','svm_select','stack','diagnosis','compare']
    newlabels={'data':'数据与控制','tuning':'强化调参','lr':'LR 调参','svm':'SVM 调参','rf':'森林调参','hgb':'提升树调参','svm_select':'特征选择','stack':'原 Stacking','diagnosis':'失败分析','compare':'全部对比'}
    # Keep earlier diagnosis as explicitly historical; add current common difficult cases.
    focus=['S0013','S0043','S0037'];case_rows=[]
    for sid in focus:case_rows.append([sid]+[pct(next(s for s in m[k]['subjects'] if s['subject']==sid)['accuracy']) for k in N])
    content['diagnosis']=head('调参后，共同失败者改善了吗','下表是本轮真实外层结果。展开下方存档，可回看原始失败模式与窗口序列。')+'<div class="panel">'+table(['受试者']+list(N.values()),case_rows)+'</div><div class="callout">仍使用相同窗口与真实标签，没有因为某人低分就排除。参数调整能改变边界，但无法凭空补充信号中缺失的状态信息；这一解释需要回看数据验证。</div><details><summary>第一轮失败分析与错误序列检查器（存档）</summary>'+content['diagnosis']+'</details>'
    return {k:content[k] for k in order},newlabels
