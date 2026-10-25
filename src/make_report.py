"""Build tables and figures from stage outputs: results/stage{1,2,3}/results_stage*.json."""
import json, os, sys
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else 'results'
FIG = os.path.join(R, 'figures'); os.makedirs(FIG, exist_ok=True)
s1 = json.load(open(f'{R}/stage1/results_stage1.json'))
s2p = f'{R}/stage2/results_stage2.json'
s2 = json.load(open(s2p)) if os.path.exists(s2p) else None
s3p = f'{R}/stage3/results_stage3.json'
if s2 and os.path.exists(s3p):  # stage 3 = strong-KL part of the beta sweep + long runs
    s3 = json.load(open(s3p)); s2['test'].update(s3['test']); s2['rl_curves'].update(s3['rl_curves'])

COLS = ['Recall@10', 'NDCG@10', 'MAP@10', 'gNDCG@10', 'coverage@10', 'gini@10', 'top1_share', 'jaccard_users',
        'series_share', 'kids_share', 'mean_len_min', 'mean_log_pop', 'top_genre_share', 'seen_share',
        'RM_wp', 'RM_time', 'RM_uncond', 'RM_ret', 'judge_wp']


def table(d):
    df = pd.DataFrame(d).T
    return df[[c for c in COLS if c in df.columns]]


out = []
t1 = table(s1['test']); out.append('## Stage 1 (test week)\n\n' + t1.round(4).to_markdown())
rm = s1['rm_eval']
rows = {k: v for k, v in rm.items() if isinstance(v, dict) and 'MAE' in v}
out.append('## Reward models (test week pairs)\n\n' + pd.DataFrame(rows).T.round(4).to_markdown())
out.append('```\n' + json.dumps({k: v for k, v in rm.items() if k not in rows}, indent=1) + '\n```')
out.append('SFT curve:\n\n' + pd.DataFrame(s1['sft_curve']).round(4).to_markdown(index=False))

if s2:
    t2 = table(s2['test']); out.append('## RL (test week)\n\n' + t2.round(4).to_markdown())
    rr = table(s2['rerank']); out.append('## Re-ranking baseline (test week)\n\n' + rr.round(4).to_markdown())
    # Pareto: NDCG vs own reward, RL (beta sweep) vs re-ranking (lambda sweep)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    base = s2['test']['SFT']
    for ax, nm in zip(axes, ['wp', 'time', 'uncond', 'ret']):
        rl = t2[t2.index.str.startswith(f'RL[{nm}] beta=')]
        rl = rl[~rl.index.str.contains('no-seen|ILD|steps')]
        rl = rl.assign(b=[float(i.split('beta=')[1]) for i in rl.index]).sort_values('b')
        lg = t2[t2.index.str.startswith(f'RL[{nm}] beta=') & t2.index.str.contains('steps')]
        if len(lg): ax.plot(lg[f'RM_{nm}'], lg['NDCG@10'], 'D', color='tab:red', label='RL 1500 steps')
        for i, r in lg.iterrows(): ax.annotate(i.split('beta=')[1], (r[f'RM_{nm}'], r['NDCG@10']), fontsize=7, color='tab:red')
        rk = rr[rr.index.str.startswith(f'rerank[{nm}]')]
        ax.plot(rl[f'RM_{nm}'], rl['NDCG@10'], 'o-', label='RL (β sweep)')
        for i, r in rl.iterrows(): ax.annotate(i.split('beta=')[1], (r[f'RM_{nm}'], r['NDCG@10']), fontsize=8)
        ax.plot(rk[f'RM_{nm}'], rk['NDCG@10'], 's--', label='re-rank (λ sweep)')
        for i, r in rk.iterrows(): ax.annotate('λ' + i.split('lam=')[1], (r[f'RM_{nm}'], r['NDCG@10']), fontsize=8)
        ax.plot(base[f'RM_{nm}'], base['NDCG@10'], 'k*', ms=14, label='SFT')
        ax.set_xlabel(f'mean RM_{nm} of top-10'); ax.set_ylabel('NDCG@10'); ax.set_title(f'reward = {nm}'); ax.grid(alpha=.3)
    axes[0].legend(); plt.tight_layout(); plt.savefig(f'{FIG}/pareto.png', dpi=120); plt.close()
    # behaviour shift at beta=0.1
    sel = ['SFT'] + [f'RL[{n}] beta=0.1' for n in ['wp', 'time', 'uncond', 'ret']]
    sel = [s for s in sel if s in t2.index]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, c in zip(axes, ['series_share', 'mean_len_min', 'mean_log_pop', 'coverage@10', 'judge_wp']):
        t2.loc[sel, c].plot.bar(ax=ax); ax.set_title(c); ax.tick_params(axis='x', rotation=60)
    plt.tight_layout(); plt.savefig(f'{FIG}/behaviour.png', dpi=120); plt.close()
    # multistep
    ms = s2['multistep']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for nm, d in ms.items():
        st = range(1, len(d['hit_autoregressive_step']) + 1)
        axes[0].plot(st, d['hit_autoregressive_step'], 'o-', label=f'{nm} autoregr.')
        axes[0].plot(st, d['hit_oneshot_rank'], 'x--', label=f'{nm} one-shot')
        axes[1].plot(st, d['KL_drift_vs_real_history'], 'o-', label=nm)
        axes[2].plot(st, d['RM_wp_synthetic_ctx'], 'o-', label=f'{nm} synthetic ctx')
        axes[2].plot(st, d['RM_wp_real_ctx'], 'x--', label=f'{nm} real ctx')
    axes[0].set_title('hit rate of k-th item'); axes[1].set_title('KL(π(·|h+g≤k) || π(·|h))'); axes[2].set_title('RM wp of k-th item')
    for a in axes: a.set_xlabel('step k'); a.legend(fontsize=7); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{FIG}/multistep.png', dpi=120); plt.close()
    out.append('## Multi-step\n\n' + pd.DataFrame(ms).T.to_markdown())
    # RL curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tag, c in s2['rl_curves'].items():
        if tag.startswith('RL[wp]') and 'no-seen' not in tag:
            df = pd.DataFrame(c); axes[0].plot(df.step, df.R_sampled, label=tag); axes[1].plot(df.step, df.KL, label=tag)
    axes[0].set_title('sampled slate reward (z)'); axes[1].set_title('KL to SFT'); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{FIG}/rl_curves.png', dpi=120); plt.close()

open(f'{R}/tables.md', 'w', encoding='utf-8').write('\n\n'.join(out))
print('\n\n'.join(out))
