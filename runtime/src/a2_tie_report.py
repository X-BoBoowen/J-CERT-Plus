"""Build the final Chinese research decision from frozen revision evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write():
    s = json.loads((ROOT/'reports/A2_TIE_RETEST_SUMMARY.json').read_text())
    audit = json.loads((ROOT/'review/TIE_FINAL_ACCOUNTING_AUDIT.json').read_text())
    witness = json.loads((ROOT/'reports/TIE_TIMED_WITNESS_VERIFICATION.json').read_text())
    replay = json.loads((ROOT/'reports/TIE_TIMED_OTC_FULL_REPLAY.json').read_text())
    fixed = json.loads((ROOT/'reports/TIE_FIXED_STATE_REPLAY.json').read_text())
    check = json.loads((ROOT/'reports/TIE_FIXED_INDEPENDENT_VERIFICATION.json').read_text())
    assert s['all_inputs_complete'] and audit['status'] == witness['status'] == replay['status'] == 'passed'
    totals = s['overall']['full_scope_service_totals_seconds']
    process = s['overall']['full_scope_process_totals_seconds']
    families = {'slow':'缓变','independent':'独立','signed-neighbor':'图相关'}
    lines = [
        'J-CERT+ A2：精确平局修订、固定输入复测与论文定位',
        '2026-11-25｜独立修订分支 research/a2-exact-ties-20261125',
        '',
        '研究判断',
        'A2 已真正免除 M 级完整候选目录的预付准备，并在本次全部固定输入上明显改善强 D-all 与分块 Stream-all 的完整服务成本；应纳入核心系统叙事。共同比较层的单节点交换证书补齐了本轮已定位的真实平局，原冻结确认仍为 failed。这里报告的是修订实现在已诊断固定输入上的复测，不是新的未见输入确认。',
        '',
        f'完整结构准备＋Q4 成本汇总：D {totals["D"]:.6f} 秒，S {totals["S"]:.6f} 秒，H {totals["H"]:.6f} 秒。固定清单下 H 相对 D 加速 {totals["D"]/totals["H"]:.4f} 倍、节省 {(1-totals["H"]/totals["D"])*100:.3f}%；相对 S 加速 {totals["S"]/totals["H"]:.4f} 倍、节省 {(1-totals["H"]/totals["S"])*100:.3f}%。',
        '全部十二单元、三臂、三次重复共108个 worker 完成；432次请求均完成15次提交，共6480次提交。没有 unresolved、超时或内存触限。这里仅有48条不同的固定意见请求，方法与计时重复不增加独立输入样本。',
        '',
        '一、完整成本先于归因',
        '时间列为三次重复的中位数，单位秒；加速列为同单元三次完整配对块的时间和之比，故不必等于两个中位数直接相除。C1/C2 是原 confirmation-1/2 标签，不表示本次有两轮新确认。完整重复范围保存在 A2_TIE_RETEST_SUMMARY.json 与 A2_TIE_RETEST_RESULTS.txt；图上的范围是 min–max，不是统计置信区间。',
        '图 | 意见族 | 原标签 | D-all | Stream-all | Implicit-all | D/H | S/H',
    ]
    for cell in s['cells']:
        costs = cell['methods']; ratios = cell['comparisons']
        label = 'C'+cell['seed_label'].rsplit('-',1)[1]
        lines.append(f'{cell["graph"]} | {families[cell["stream"]]} | {label} | '+
                     ' | '.join(f'{costs[m]["median_seconds"]:.3f}' for m in 'DSH')+
                     f' | {ratios["H_over_D"]["inverse_speedup_reference_over_H"]:.3f} | {ratios["H_over_S"]["inverse_speedup_reference_over_H"]:.3f}')
    lines += [
        '',
        'D 在全部36个配对重复块中都比 S 快，所以更快强参照始终是 D。H 对 D、S 的36个配对块均严格更快；最大单次 H/D 为 '+f'{audit["per_repeat_ratios"]["D"]["maximum"]:.6f}'+
        '。十二单元的配对和加速范围为 D/H 2.660–4.244 倍，S/H 10.320–17.190 倍。总和仅描述这份固定混合清单，不外推为输入分布上的期望收益。',
        f'更宽进程墙时单列：D {process["D"]:.3f} 秒，S {process["S"]:.3f} 秒，H {process["H"]:.3f} 秒；对应 D/H {process["D"]/process["H"]:.3f} 倍、S/H {process["S"]/process["H"]:.3f} 倍。它包括启动、检查和检查点等服务边界外开销。独立 checker 不包含在算法主时间中。',
        '',
        '二、省掉的工作与仍然支付的工作',
        '共同目录为所有根非边上的正边插入（ω=1/16），加根图负边的自然权重翻转。Alpha：n=3783、M=7,140,884；OTC：n=5881、M=17,271,859。两图保留自然边权；三类输入均为生成意见。每请求从同一根打开，Gram 初始为空，答案缓存关闭，δ=0，最多15次连续提交。',
        'D 保留强根索引、漂移发现和固定结构复用；S 使用有界分块向量化完整扫描及付费种子；H 使用当前版本联合前沿。H/S 没有完整非边表、M 项权重或 cap、候选 incidence、M 项 visited，也没有免费全局 top-64。新平局 helper 只读当前稀疏算子、精确意见及实际竞争者，不重新引入 M 级结构。',
        '以下工作量为每图18个完整 Q4 worker 的均值。端点检查是跨版本的检查次数，可重复；Gram 数、响应块数、发出候选和 heap pop 分开计数。',
        '图/方法 | 端点检查/Q4 | Gram/Q4 | 响应块/Q4 | 当前前沿建立/Q4 | 进程峰值RSS中位数MiB（重复/输入范围）',
    ]
    for graph in s['graphs']:
        for method in 'DSH':
            rec=graph['methods'][method]; work=rec['work']; rss=rec['process_peak_rss_bytes']
            val=lambda key:work[key]['complete_workload_total']/rec['planned_workers']
            lines.append(f'{graph["name"]}/{method} | {val("numeric.endpoint_checked"):,.1f} | '+
                         f'{val("numeric.gram_materializations"):.1f} | {val("numeric.response_blocks"):.1f} | '+
                         f'{val("access.frontier_rebuilds"):.0f} | {rss["median"]/2**20:.2f} '+
                         f'({rss["minimum"]/2**20:.2f}–{rss["maximum"]/2**20:.2f})')
    lines += [
        '',
        '按图汇总，H 与 D 的平均 Gram 数和响应块数相同。主要收益对应候选前端与发现工作，而非放宽评分契约或减少保证。S 已经避免完整目录常驻数组，其 RSS 与 H 接近；H 在 S 之上减少了完整扫描，这使强流式比较具有决定作用。',
        'H 每完整 Q4 实际建立60次当前前沿，全部计费。H 每 Q4 的合法插入发出数、翻转发出数、raw heap pop 均值分别为：Alpha 659.33、247.33、1842.50；OTC 101.83、314.17、763.83。它们既不是不同候选数，也不是响应数。另有 warm 候选复查；上表端点检查已包含其工作。全量工作计数、RHS 分解、采样RSS与分配载荷在机器汇总中保留。',
        '前沿的 O(n+F) 空间不代表整个引擎空间。稀疏因子填充、状态、low-rank 传递数据、实际 Gram 缓存、Python 集合和日志都另计。缓存最坏可以随访问增长到目录规模。RSS 为进程生命周期高水位，表中区间覆盖每图18个 worker，并非置信区间；外部采样峰值另存。',
        '',
        '三、修订只补齐共同比较能力',
        '新增 src/exact_ties.py；原运行时集合中仅修改 weighted_adapter.py 的 resolve 与 a2_session.py 的当前状态绑定、证书记录和计数。独立 checker 与复测/分析工具另行扩展。H 的 heap、D 的索引、S 的扫描、包络、意见、精度和容差未改。旧五参数 resolver 调用仍保持原行为，A2 三臂均传入当前精确上下文。',
        '对实际重叠候选 e、f，第一版只尝试单个节点交换 P。它检查当前 A 与精确意见 s，两个候选的当前合法性、类型、整数权重和完整更新关系，证明 PAPᵀ=A、Ps=s、P(A+Δe)Pᵀ=A+Δf。因此两个 child 二次型及增益严格相等。证书绑定当前父版本、算子和意见摘要、端点、操作、权重与单位；跨提交重新验证。',
        '只有全部未决竞争者都被处理，才按规范 ID 选赢家；接受仍要求严格正的数值下界。等值关系允许区间相交：保留正下界，使用真实上界的最小值，不把下界充当上界，不制造点分数。一个对的见证不能跳过第三个未解决竞争者。单交换无法覆盖的等值仍可 unresolved，未加入一般图同构或精度调度。',
        '规范赢家变化时，提交仍通过原 response 的候选身份检查取得赢家自身响应/枢轴。固定重放在 Alpha/C2/q2 第12步实际出现 10085176→10084922，交换节点3227/3481；对应提交及后续全目录检查均通过。',
        '',
        '四、限定诊断与独立验证的实际范围',
        '先在未改原实现上只重放交接的 OTC/C1/q4，重现第7步退出；前六步、当前父算子、实际候选13241964/13241965及精确交换3833/3834全部匹配。没有在这次原版诊断中接受第七个编辑。',
        '然后对七个确认失败状态、三个开发失败状态各重放一次 H 请求。十个请求均保持原成功前缀并完成15步；实际发现14次重叠比较，发出14份精确见证，没有未证明竞争者。十个固定请求的每个决策均由独立全目录 checker 复证：150个决策、1,223,096,175个候选—决策项；耗时 '+f'{check["wall_seconds"]:.3f}'+ ' 秒，进程峰值RSS约73.17 MiB。',
        '针对性测试108项通过，覆盖意见/权重/操作改变、当前图对称性破坏、过期证书、部分多竞争者、规范赢家响应、零/负/正收益等。保留了单交换不能解决的确定性大图未决测试。没有重跑整套A1或不受影响的历史性能实验。',
        '实际计时轨迹另经独立重建全部432条请求链、6480个父状态，核验90份见证观察，去重为10个当前状态关系；有9次规范赢家切换观察，去重为1个状态。此项只证明当前绑定、结构关系和记录算术，不声称对全部432请求做了独立全目录求值。',
        '按复测前固定选择，对实际计时的 H/repeat0/OTC/C1/q4 再做完整15步全目录复证：259,077,780个候选—决策项、91个独立RHS、1份结构见证；耗时 '+f'{replay["wall_seconds"]:.3f}'+ ' 秒。所有正收益与竞争责任都独立验证。大图发布区间检查沿用与独立包络相交的一致性口径，不扩大为整个发布区间的独立包含性证明。',
        '还从原始JSON独立复算总成本、逐重复比值和原有成功前缀：432条请求中原先已接受的5769个决策，其赢家、父子版本、low和high全部保持一致。运行时源摘要与冻结一致；原十二份NPZ及其清单逐字节复用。',
        '',
        '五、冻结边界与成本表述',
        '原 A2 确认及 failed 判定保留于 reports/A2_CONFIRMATION_JUDGMENT.json，原提交365bc8e不动。共同平局实现冻结于531522d，原输入复测绑定冻结于9de16a7。新复测只执行一次，108个串行轮换作业，无重试、换种子或调参；单worker180秒、2GiB，批次3600秒；实际批次 '+f'{audit["batch_time_fields"]["batch_wall_seconds"]:.3f}'+ ' 秒。A1继续关闭，R15不改。',
        '既有前沿操作界 O(n log n+(K+P+mG+b)log n)+O(F log F) 针对给定排除集合；K包含严格筛查后被丢弃的合法发出候选，P为已处理/warm跳过的合法pop。服务层反复 hot|chosen 的集合构造、warm排序/筛查、当前数值状态、响应、Gram传递、平局验证和日志另列。所有实际版本重复付费，不能把一次 n log n 写成整条轨迹的代价。详细修订见 TIE_EQUALITY_PROOF.txt 的费用部分；实际墙时已经包含这些工作。',
        '',
        '六、论文应如何接住 A2',
        '建议把 A2 纳入主要系统贡献：围绕“为了证明当前全目录决定正确，必须计算和保存什么”展开。一般目录保留强显式路径；在结构化目录中，未产生候选由联合前沿覆盖，竞争者才取得局部响应，已有 Gram 随版本传递，每次提交重建当前覆盖，数值区间与可核验等值关系共同完成规范选择。',
        '排序、堆、阈值停止、Sherman–Morrison 和节点交换等值证明均不登记为本项目新发明。TA 和 Rank-Join 已提供相关经典基础。新增系统价值在于确实删除全目录依赖，将混合合法编辑、当前认证状态、按需响应、连续提交后的覆盖和完整成本结合，并相对同域强 D/S 给出可复算结果。缺少实例最优定理不自动否定系统贡献；本报告也不主张实例最优、普遍次线性查询或Best Paper保证。',
        '适用范围必须贴近结论：固定根服务、统一插入权重、稀疏自然权重翻转、两张完整图的生成意见、Q4/b15。没有外推到two-hop、任意插入权重/过滤、外部动态图或一般精确平局。原失败与修订复测不是同一个实验判定。A1仅保留为有范围的稳健实现机制。',
        '本轮算法工作到此结束。交付一页“主张—证据—章节目标”与三张候选图，进入既有证据的论文整合；不自动追加新控制器、精度路线、输入种子或确认批次。',
        '',
        '主要证据文件',
        'reports/A2_TIE_RETEST_SUMMARY.json；reports/A2_TIE_RETEST_RESULTS.txt；reports/TIE_RETEST_ANALYSIS.json。',
        'reports/TIE_FIXED_STATE_INVENTORY.json；reports/TIE_FIXED_STATE_REPLAY.json；reports/TIE_FIXED_INDEPENDENT_VERIFICATION.json。',
        'reports/TIE_TIMED_WITNESS_VERIFICATION.json；reports/TIE_TIMED_OTC_FULL_REPLAY.json；review/TIE_FINAL_ACCOUNTING_AUDIT.json。',
        'reports/TIE_EQUALITY_PROOF.txt；review/TIE_TARGETED_TESTS.txt；review/TIE_REVISION_CODE_REVIEW.txt。',
        'protocol/A2_TIE_RETEST.json；protocol/A2_TIE_RETEST_ADMISSION.json；protocol/A2_TIE_RETEST_INPUTS.json；raw/a2_tie_retest/。',
        '',
        '既有研究来源',
        'Fagin, Lotem, Naor. Optimal aggregation algorithms for middleware (2001). https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware',
        'Ilyas, Aref, Elmagarmid. Supporting Top-k Join Queries in Relational Databases (2003), §4.2. https://www.cs.purdue.edu/homes/aref/papers/vldb03_TopKJoin.pdf',
        '',
    ]
    (ROOT/'reports/A2_TIE_REVISION_REPORT_ZH.txt').write_text('\n'.join(lines))
    judgment = dict(status='completed_fixed_input_revision_retest',
        research_decision='include_A2_in_core_system_narrative_with_explicit_domain',
        original_confirmation_status='failed', unseen_confirmation=False,
        original_confirmation_reclassified=False, further_algorithm_work='none in this round',
        runtime_commit='531522d', retest_binding_commit='9de16a7',
        all_12_cells_complete=True, completed_workers=108, completed_requests=432,
        accepted_edits=6480, paired_repeats_faster_than_D=36, paired_repeats_faster_than_S=36,
        service_totals_seconds=totals, process_totals_seconds=process,
        speedups={r:totals[r]/totals['H'] for r in ('D','S')},
        certificates=dict(timed_observations=90,unique_current_relations=10,
                          canonical_switch_observations=9,unique_canonical_switch_states=1),
        independent_scope=dict(fixed_complete_requests=10,fixed_decisions=150,
                               all_timed_parent_chains=432,all_timed_witness_observations=90,
                               timed_full_catalog_requests=1,timed_full_catalog_decisions=15,
                               published_large_intervals='intersection consistency only'),
        scope='fixed-root all nonedges omega1/16 plus natural-weight root negative flips; Alpha/OTC; generated opinions; Q4; b15',
        novelty='scoped system integration and measured cost removal; no new claim to classic threshold/enumeration/equality algebra or instance optimality')
    (ROOT/'reports/A2_TIE_REVISION_JUDGMENT.json').write_text(json.dumps(judgment,indent=2)+'\n')


if __name__ == '__main__':
    write()
