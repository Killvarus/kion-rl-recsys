"""KION: GPT-2 next-item recommender -> reward models -> RL fine-tuning -> analysis.

One file, three stages:
  STAGE=1 : data prep, baselines, GPT-2 SFT, neural reward model (RM-A), CatBoost judge (RM-B)
  STAGE=2 : RL fine-tuning (RLOO policy gradient over Plackett-Luce slates + exact KL to SFT),
            re-ranking baseline, multi-step generation analysis
  STAGE=3 : strong-KL part of the beta sweep and long RL runs
  STAGE=4 : item-relative rewards, negative sampling of the unconditional head, multi-step (trajectory) RL,
            off-policy REINFORCE on logged feedback and positive-only fine-tuning
Env: KION_DIR (dir containing data_original.zip or interactions.csv/items.csv), OUT_DIR, SMOKE (n users).
"""
import os, glob, json, time, zipfile, copy, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

STAGE = os.environ.get('STAGE', globals().get('STAGE', '1'))
SMOKE = int(os.environ.get('SMOKE', '0'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IN = os.environ.get('KION_DIR', './data')
OUT = os.environ.get('OUT_DIR', './out')
PREV = os.environ.get('PREV_DIR', './results/stage1')  # stage-1 outputs used by stages 2-4
os.makedirs(OUT, exist_ok=True)

T_VAL, T_TEST = pd.Timestamp('2021-08-09'), pd.Timestamp('2021-08-16')
MAXLEN, K, PAD, SEP, OFF = 50, 10, 0, 1, 2
MINCNT = 2 if SMOKE else 10
NEVAL = 2000 if SMOKE else 20000
NEG = -1e4
D_MODEL, N_LAYER, N_HEAD = 128, 2, 4
SFT_EPOCHS = 1 if SMOKE else 12
RM_EPOCHS = 1 if SMOKE else 3
RL_STEPS = 6 if SMOKE else 400
REWARDS = ['wp', 'time', 'uncond', 'ret']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def seed_all(s=0):
    np.random.seed(s); torch.manual_seed(s)


def find(name, root):
    hits = [h for h in glob.glob(os.path.join(root, '**', name), recursive=True) if '__MACOSX' not in h]
    return hits[0] if hits else None


# ============================== DATA ==============================
def load_raw():
    p, pi = find('interactions.csv', IN), find('items.csv', IN)
    if p and pi and os.path.getsize(p) > 100e6:
        return pd.read_csv(p), pd.read_csv(pi)
    z = zipfile.ZipFile(find('data_original.zip', IN))  # NB: top-level interactions.csv.zip of the repo is truncated
    return pd.read_csv(z.open('data_original/interactions.csv')), pd.read_csv(z.open('data_original/items.csv'))


def prepare():
    inter, items = load_raw()
    log('raw rows', len(inter))
    inter = inter.dropna(subset=['watched_pct', 'total_dur']).copy()
    inter['dt'] = pd.to_datetime(inter['last_watch_dt'])
    if SMOKE:
        u = inter.user_id.unique(); keep = np.random.RandomState(0).choice(u, min(len(u), SMOKE), replace=False)
        inter = inter[inter.user_id.isin(keep)]
    # within-day order is unknown (date granularity) -> fixed random tie-break
    inter['rnd'] = np.random.RandomState(1).rand(len(inter))
    inter = inter.sort_values(['user_id', 'dt', 'rnd']).reset_index(drop=True)
    tr = inter[inter.dt < T_VAL]
    cnt = tr.item_id.value_counts()
    vocab = np.sort(cnt[cnt >= MINCNT].index.values)
    V = len(vocab) + OFF
    item2tok = pd.Series(np.arange(OFF, V), index=vocab)
    inter['tok'] = inter.item_id.map(item2tok).fillna(-1).astype(np.int64)
    # long-term label: user is active again (another date) within 7 days after this interaction's date
    ud = inter[['user_id', 'dt']].drop_duplicates().sort_values(['user_id', 'dt'])
    ud['next_dt'] = ud.groupby('user_id').dt.shift(-1)
    inter = inter.merge(ud, on=['user_id', 'dt'], how='left')
    gap = (inter.next_dt - inter.dt).dt.days
    inter['ret7'] = (gap <= 7).astype(np.float32)
    inter['ret7_ok'] = (inter.dt <= T_VAL - pd.Timedelta(days=8)).astype(np.float32)  # label window inside train

    # ---------- item info (train statistics only) ----------
    meta = items.set_index('item_id').reindex(vocab)
    trv = tr.assign(tok=tr.item_id.map(item2tok)).dropna(subset=['tok'])
    trv['tok'] = trv.tok.astype(int)
    trv['logdur'] = np.log1p(trv.total_dur) / 10
    st = trv.groupby('tok').agg(n=('tok', 'size'), wp=('watched_pct', 'mean'), ld=('logdur', 'mean'))
    lr = trv[trv.watched_pct >= 10]
    length = (lr.total_dur / (lr.watched_pct / 100) / 60).groupby(lr.tok).median()
    info = {}
    info['tok2item'] = np.r_[[-1] * OFF, vocab]
    ser = (meta.content_type.values == 'series').astype(np.float32)
    info['series'] = np.r_[[0] * OFF, ser].astype(np.float32)
    info['kids'] = np.r_[[0] * OFF, meta.for_kids.fillna(0).values].astype(np.float32)
    Lv = np.full(V - OFF, np.nan); Lv[length.index.values - OFF] = length.values
    for s in [0, 1]:
        msk = ser == s
        fill = np.nanmedian(Lv[msk]) if np.isfinite(Lv[msk]).any() else 60.
        Lv[msk & np.isnan(Lv)] = fill
    info['length'] = np.r_[[1.] * OFF, np.clip(Lv, 1, 3000)].astype(np.float32)
    n = np.zeros(V, np.float32); n[st.index] = st.n.values; info['n_train'] = n
    wpm = np.full(V, tr.watched_pct.mean(), np.float32); wpm[st.index] = st.wp.values; info['wp_mean'] = wpm
    ldm = np.full(V, trv.logdur.mean(), np.float32); ldm[st.index] = st.ld.values; info['logdur_mean'] = ldm
    g = meta.genres.fillna('').str.split(', ')
    top_g = pd.Series([x for l in g for x in l if x]).value_counts().index[:30]
    G = np.zeros((V, len(top_g)), np.float32)
    gi = {x: i for i, x in enumerate(top_g)}
    for r, l in enumerate(g):
        for x in l:
            if x in gi: G[r + OFF, gi[x]] = 1
    info['G'] = G; info['genres'] = list(top_g)
    age = meta.age_rating.fillna(-1).values
    A = np.stack([(age == a) for a in [0, 6, 12, 16, 18]], 1).astype(np.float32)
    year = np.clip(meta.release_year.fillna(2010).values, 1950, 2022)
    feat = np.concatenate([ser[:, None], info['kids'][OFF:, None], A, ((year - 2005) / 15)[:, None],
                           ((np.log(info['length'][OFF:]) - 4) / 1.5)[:, None], G[OFF:]], 1)
    info['feat'] = np.r_[np.zeros((OFF, feat.shape[1])), feat].astype(np.float32)
    info['V'] = V

    # ---------- per-user sequences ----------
    uid = inter.user_id.values
    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]]); ends = np.r_[starts[1:], len(uid)]
    tok, wp, dur, dt = inter.tok.values, inter.watched_pct.values.astype(np.float32), inter.total_dur.values.astype(np.float32), inter.dt.values
    ret, retok = inter.ret7.values, inter.ret7_ok.values
    tv, tt = np.datetime64(T_VAL), np.datetime64(T_TEST)
    train_seqs, ev = [], {'val': [], 'test': []}
    for s, e in zip(starts, ends):
        d = dt[s:e]; t = tok[s:e]
        trm = (d < tv) & (t >= 0)
        if trm.sum() >= 2:
            idx = np.flatnonzero(trm)[-(MAXLEN + 1):]
            train_seqs.append((t[idx], wp[s:e][idx], dur[s:e][idx], ret[s:e][idx], retok[s:e][idx]))
        for name, lo, hi in [('val', tv, tt), ('test', tt, np.datetime64('2100-01-01'))]:
            hm = (d < lo) & (t >= 0)
            tm = (d >= lo) & (d < hi)
            if hm.sum() >= 2 and tm.any():
                hidx = np.flatnonzero(hm)[-MAXLEN:]
                ev[name].append(dict(uid=uid[s], hist=t[hidx], hwp=wp[s:e][hidx], hdur=dur[s:e][hidx],
                                     tgt=t[tm], twp=wp[s:e][tm], tret=ret[s:e][tm], tdt=d[tm]))
    rs = np.random.RandomState(2)
    for k in ev:
        sel = rs.choice(len(ev[k]), min(NEVAL, len(ev[k])), replace=False)
        log(k, 'eligible users', len(ev[k]), '-> sampled', len(sel))
        ev[k] = [ev[k][i] for i in sel]
    # popularity baseline lists (last 14 days before each cutoff, in-vocab)
    pop = {}
    for k, cut in [('val', T_VAL), ('test', T_TEST)]:
        w = inter[(inter.dt < cut) & (inter.dt >= cut - pd.Timedelta(days=14)) & (inter.tok >= 0)]
        pop[k] = w.tok.value_counts().index.values[:300]
    log('V', V, 'train seqs', len(train_seqs))
    stats = dict(rows=len(inter), users=int(inter.user_id.nunique()), V=V, n_train_seqs=len(train_seqs),
                 n_val_users=len(ev['val']), n_test_users=len(ev['test']))
    return train_seqs, ev, info, pop, stats


def pad(seqs, extra=None):
    L = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), L), np.int64); m = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs): x[i, :len(s)] = s; m[i, :len(s)] = 1
    return torch.from_numpy(x).to(DEV), torch.from_numpy(m).to(DEV)


# ============================== MODELS ==============================
def make_gpt(V):
    cfg = GPT2Config(vocab_size=V, n_positions=MAXLEN + K + 4, n_embd=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD,
                     resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2, bos_token_id=SEP, eos_token_id=SEP, pad_token_id=PAD)
    return GPT2LMHeadModel(cfg).to(DEV)


def last_hidden(tr, x, m):
    h = tr(input_ids=x, attention_mask=m).last_hidden_state
    return h[torch.arange(len(x), device=DEV), m.sum(1) - 1]


def policy_logits(model, x, m, mask_seen=True):
    lg = model.lm_head(last_hidden(model.transformer, x, m))
    lg[:, :OFF] = NEG
    if mask_seen: lg.scatter_(1, x, NEG)
    return lg


class RewardModel(nn.Module):
    """RM-A: causal transformer over history -> user state; MLP(state, item emb, item features) -> heads:
    0 cond E[watched_pct] (BCE on pct/100), 1 cond P(watched>=80%), 2 cond E[log1p(total_dur)]/10,
    3 uncond E[watched_pct * 1{watch}] (trained with sampled negatives), 4 P(return within 7 days | watched)."""
    def __init__(self, sft, feat):
        super().__init__()
        self.tr = copy.deepcopy(sft.transformer)
        self.emb = nn.Embedding.from_pretrained(sft.transformer.wte.weight.detach().clone(), freeze=False)
        self.register_buffer('feat', torch.tensor(feat))
        d = D_MODEL; fin = 3 * d + feat.shape[1]
        self.mlp = nn.Sequential(nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU())
        self.out = nn.Linear(128, 5)

    def state_all(self, x, m): return self.tr(input_ids=x, attention_mask=m).last_hidden_state
    def state(self, x, m): return last_hidden(self.tr, x, m)

    def head(self, s, items):  # s (..., d) broadcastable to items (...)
        e = self.emb(items); f = self.feat[items]
        s = s.expand(*items.shape, s.shape[-1])
        return self.out(self.mlp(torch.cat([s, e, s * e, f], -1)))


def head_value(h, name):
    if name == 'wp': return torch.sigmoid(h[..., 0])
    if name == 'complete': return torch.sigmoid(h[..., 1])
    if name == 'time': return h[..., 2]
    if name == 'uncond': return torch.sigmoid(h[..., 3])
    if name == 'ret': return torch.sigmoid(h[..., 4])
    raise ValueError(name)


def attach_item_base(rm, info):
    """Per-item baselines (train means) for item-relative rewards: 'wp_rel', 'time_rel'."""
    rm.item_base = {'wp': torch.tensor(info['wp_mean'] / 100, device=DEV),
                    'time': torch.tensor(info['logdur_mean'], device=DEV)}
    return rm


def reward_value(rm, s, items, name):
    """Reward of `items` for user states `s`. '<head>_rel' = head prediction minus the item's train mean,
    i.e. engagement above what this item usually gets."""
    base = name[:-4] if name.endswith('_rel') else name
    v = head_value(rm.head(s, items), base)
    if name.endswith('_rel'): v = v - rm.item_base[base][items]
    return v


# ============================== EVAL ==============================
def recommend(model, hists, k=K, bs=2048, mask_seen=True):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(hists), bs):
            x, m = pad(hists[i:i + bs])
            out.append(policy_logits(model, x, m, mask_seen).topk(k, -1).indices.cpu().numpy())
    return np.concatenate(out)


def rec_pop(poplist, hists, k=K):
    out = []
    for h in hists:
        hs = set(h.tolist()); out.append([t for t in poplist if t not in hs][:k])
    return np.array(out)


def rm_states(rm, hists, bs=2048):
    rm.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(hists), bs):
            x, m = pad(hists[i:i + bs]); out.append(rm.state(x, m))
    return torch.cat(out)


def rm_scores(rm, S, recs, name, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(S), bs):
            r = torch.from_numpy(np.asarray(recs[i:i + bs])).to(DEV)
            out.append(reward_value(rm, S[i:i + bs, None, :], r, name).cpu().numpy())
    return np.concatenate(out)


def gini(c):
    c = np.sort(c.astype(np.float64)); n = len(c)
    return float(((2 * np.arange(1, n + 1) - n - 1) * c).sum() / (n * c.sum() + 1e-12))


def metrics(recs, evs, info, rm=None, S=None, judge=None):
    disc = 1 / np.log2(np.arange(2, K + 2))
    R, ND, MP, GN, HR, SEEN = [], [], [], [], [], []
    for u, e in enumerate(evs):
        rec = recs[u]; tgt = e['tgt']; n = len(tgt)
        valid = tgt[tgt >= 0]
        hit = np.isin(rec, valid)
        R.append(hit.sum() / n)
        ND.append((hit * disc).sum() / disc[:min(n, K)].sum())
        prec = np.cumsum(hit) / np.arange(1, K + 1); MP.append((prec * hit).sum() / min(n, K))
        gmap = dict(zip(tgt.tolist(), (e['twp'] / 100).tolist()))
        gains = np.array([gmap.get(t, 0.) if h else 0. for t, h in zip(rec.tolist(), hit)])
        ideal = np.sort(e['twp'] / 100)[::-1][:K]
        idg = (ideal * disc[:len(ideal)]).sum()
        GN.append((gains * disc).sum() / idg if idg > 0 else 0.)
        HR.append(hit.any()); SEEN.append(np.isin(rec, e['hist']).mean())
    flat = recs.ravel(); V = info['V']
    cnt = np.bincount(flat, minlength=V)[OFF:]
    rs = np.random.RandomState(0); a, b = rs.randint(0, len(recs), 3000), rs.randint(0, len(recs), 3000)
    jac = np.mean([len(set(recs[i]) & set(recs[j])) / len(set(recs[i]) | set(recs[j])) for i, j in zip(a, b) if i != j])
    G = info['G'][recs]  # (N,K,g) genre concentration within a list
    gshare = (G.sum(1).max(1) / np.maximum(G.sum((1, 2)), 1)).mean()
    res = {'Recall@10': np.mean(R), 'NDCG@10': np.mean(ND), 'MAP@10': np.mean(MP), 'gNDCG@10': np.mean(GN),
           'HitRate@10': np.mean(HR), 'coverage@10': float((cnt > 0).mean()), 'gini@10': gini(cnt),
           'top1_share': float(np.bincount(recs[:, 0]).max() / len(recs)), 'jaccard_users': float(jac),
           'series_share': float(info['series'][flat].mean()), 'kids_share': float(info['kids'][flat].mean()),
           'mean_len_min': float(info['length'][flat].mean()), 'mean_log_pop': float(np.log1p(info['n_train'][flat]).mean()),
           'top_genre_share': float(gshare), 'seen_share': float(np.mean(SEEN))}
    if rm is not None:
        for nm in REWARDS + ['complete'] + (['wp_rel', 'time_rel'] if hasattr(rm, 'item_base') else []):
            res['RM_' + nm] = float(rm_scores(rm, S, recs, nm).mean())
    if judge is not None:
        jw = judge(evs, recs); res['judge_wp'] = float(jw.mean())
        res['judge_wp_rel'] = float((jw - info['wp_mean'][recs] / 100).mean())
    return {k: float(v) for k, v in res.items()}


# ============================== JUDGE (RM-B) ==============================
def user_table(evs, info):
    G, rows = info['G'], []
    for e in evs:
        h = e['hist']
        prof = G[h].mean(0)
        rows.append(np.r_[len(h), e['hwp'].mean(), np.log1p(e['hdur']).mean(), info['series'][h].mean(),
                          info['kids'][h].mean(), np.log1p(info['n_train'][h]).mean(), prof])
    return np.array(rows, np.float32)


def pair_features(U, toks, uidx, info):
    ng = len(info['genres']); prof = U[uidx, -ng:]; Gi = info['G'][toks]
    aff = (prof * Gi).sum(1) / (np.linalg.norm(prof, axis=1) * np.linalg.norm(Gi, axis=1) + 1e-6)
    it = np.stack([info['series'][toks], info['kids'][toks], np.log(info['length'][toks]), np.log1p(info['n_train'][toks]),
                   info['wp_mean'][toks], info['feat'][toks, 2:8].argmax(1)], 1)
    return np.concatenate([U[uidx, :-ng], it, aff[:, None], (U[uidx, 3] * info['series'][toks])[:, None]], 1)


def judge_pairs(evs, info):
    U = user_table(evs, info); X, y, g = [], [], []
    for u, e in enumerate(evs):
        msk = e['tgt'] >= 0
        if msk.any():
            X.append(pair_features(U, e['tgt'][msk], np.full(msk.sum(), u), info)); y.append(e['twp'][msk] / 100); g.append(np.full(msk.sum(), u))
    return np.concatenate(X), np.concatenate(y), np.concatenate(g)


def make_judge_fn(model, info):
    def fn(evs, recs):
        U = user_table(evs, info); N = len(recs)
        X = pair_features(U, recs.ravel(), np.repeat(np.arange(N), recs.shape[1]), info)
        return model.predict(X).reshape(N, -1)
    return fn


# ============================== STAGE 1 ==============================
def train_sft(train_seqs, ev_val, info, results):
    model = make_gpt(info['V'])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    seqs = [s[0] for s in train_seqs]; bs = 256; best, best_state, bad = -1, None, 0
    hv = [e['hist'] for e in ev_val]; curve = []
    for ep in range(SFT_EPOCHS):
        model.train(); perm = np.random.permutation(len(seqs)); tot = 0; t0 = time.time()
        for i in range(0, len(perm), bs):
            x, m = pad([seqs[j] for j in perm[i:i + bs]])
            lab = x.masked_fill(m == 0, -100)
            loss = model(input_ids=x, attention_mask=m, labels=lab).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        r = metrics(recommend(model, hv), ev_val, info)
        curve.append({'epoch': ep, 'loss': tot / (len(perm) / bs), 'val_NDCG@10': r['NDCG@10'], 'val_Recall@10': r['Recall@10']})
        log('SFT', curve[-1], f'{time.time() - t0:.0f}s')
        if r['NDCG@10'] > best: best, best_state, bad = r['NDCG@10'], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 2: break
    model.load_state_dict(best_state); results['sft_curve'] = curve
    return model


def train_rm(sft, train_seqs, info, results, neg_mode='pop'):
    """neg_mode for the unconditional head: 'pop' (popularity-proportional), 'uniform', 'mix' (half/half)."""
    rm = RewardModel(sft, info['feat']).to(DEV)
    opt = torch.optim.AdamW(rm.parameters(), lr=5e-4, weight_decay=0.01)
    V = info['V']; pop = torch.tensor(info['n_train'] / info['n_train'].sum(), device=DEV).clamp_min(1e-12)
    data = train_seqs; bs = 256; NNEG = 4
    for ep in range(RM_EPOCHS):
        rm.train(); perm = np.random.permutation(len(data)); tot = 0
        for i in range(0, len(perm), bs):
            b = [data[j] for j in perm[i:i + bs]]
            x, m = pad([s[0] for s in b])
            L = x.shape[1]
            def padf(k):
                a = np.zeros((len(b), L), np.float32)
                for r, s in enumerate(b): a[r, :len(s[k])] = s[k]
                return torch.from_numpy(a).to(DEV)
            wp, dur, ret, rok = padf(1) / 100, padf(2), padf(3), padf(4)
            H = rm.state_all(x, m)[:, :-1]  # state before item t
            items, msk = x[:, 1:], m[:, 1:].float()
            h = rm.head(H, items)
            wpt = wp[:, 1:]
            l_wp = F.binary_cross_entropy_with_logits(h[..., 0], wpt, reduction='none')
            l_cp = F.binary_cross_entropy_with_logits(h[..., 1], (wpt >= 0.8).float(), reduction='none')
            l_du = (h[..., 2] - torch.log1p(dur[:, 1:]) / 10) ** 2
            l_un = F.binary_cross_entropy_with_logits(h[..., 3], wpt, reduction='none')
            n_pop = {'pop': NNEG, 'uniform': 0, 'mix': NNEG // 2}[neg_mode]
            parts = []
            if n_pop: parts.append(torch.multinomial(pop, items.numel() * n_pop, replacement=True).view(*items.shape, n_pop))
            if NNEG - n_pop: parts.append(torch.randint(OFF, V, (*items.shape, NNEG - n_pop), device=DEV))
            neg = torch.cat(parts, -1)
            hn = rm.head(H[:, :, None, :], neg)
            l_un = l_un + F.binary_cross_entropy_with_logits(hn[..., 3], torch.zeros_like(hn[..., 3]), reduction='none').sum(-1)
            rm_ = rok[:, 1:] * msk
            l_rt = F.binary_cross_entropy_with_logits(h[..., 4], ret[:, 1:], reduction='none') * rm_
            loss = ((l_wp + l_cp + l_du + l_un) * msk).sum() / msk.sum() + l_rt.sum() / rm_.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0); opt.step(); tot += loss.item()
        log('RM epoch', ep, neg_mode, 'loss', tot / (len(perm) / bs))
    rm.eval()
    return rm


def eval_rm(rm, judge_model, ev_test, ev_val, info, train_mean_wp, results):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    S = rm_states(rm, [e['hist'] for e in ev_test])
    P, Y, IM, UM, U, JD = [], [], [], [], [], []
    Xj, yj, gj = judge_pairs(ev_test, info); jpred = judge_model.predict(Xj)
    for u, e in enumerate(ev_test):
        msk = e['tgt'] >= 0
        if not msk.any(): continue
        t = e['tgt'][msk]
        with torch.no_grad():
            h = rm.head(S[u:u + 1], torch.from_numpy(t)[None].to(DEV))[0]
        P.append(np.stack([head_value(h, 'wp').cpu().numpy(), head_value(h, 'complete').cpu().numpy()], 1))
        Y.append(e['twp'][msk] / 100); IM.append(info['wp_mean'][t] / 100); UM.append(np.full(len(t), e['hwp'].mean() / 100)); U.append(np.full(len(t), u))
    P, Y, IM, UM, U = np.concatenate(P), np.concatenate(Y), np.concatenate(IM), np.concatenate(UM), np.concatenate(U)
    comp = (Y >= 0.8).astype(int)
    def sp(pred):
        vals = []
        for u in np.unique(U):
            mk = U == u
            if mk.sum() >= 3 and np.std(Y[mk]) > 0 and np.std(pred[mk]) > 0: vals.append(spearmanr(pred[mk], Y[mk])[0])
        return float(np.mean(vals))
    out = {}
    for nm, pr in [('global_mean', np.full_like(Y, train_mean_wp / 100)), ('item_mean', IM), ('user_mean', UM),
                   ('RM-A (neural)', P[:, 0]), ('RM-B judge (CatBoost)', jpred)]:
        out[nm] = {'MAE': float(np.abs(pr - Y).mean()), 'RMSE': float(np.sqrt(((pr - Y) ** 2).mean())),
                   'AUC_complete': float(roc_auc_score(comp, pr)) if np.std(pr) > 0 else 0.5,
                   'within_user_spearman': sp(pr) if np.std(pr) > 0 else 0.0}
    out['RM-A complete head'] = {'AUC_complete': float(roc_auc_score(comp, P[:, 1]))}
    # unconditional head: watched-in-test-week items vs popularity-sampled non-watched items
    rs = np.random.RandomState(3); popp = info['n_train'] / info['n_train'].sum()
    pos_s, neg_s, sft_like = [], [], []
    for u, e in enumerate(ev_test):
        t = e['tgt'][e['tgt'] >= 0]
        if not len(t): continue
        ng = rs.choice(info['V'], len(t), p=popp)
        with torch.no_grad():
            hp = rm.head(S[u:u + 1], torch.from_numpy(t)[None].to(DEV))[0]; hn = rm.head(S[u:u + 1], torch.from_numpy(ng)[None].to(DEV))[0]
        pos_s.append(head_value(hp, 'uncond').cpu().numpy()); neg_s.append(head_value(hn, 'uncond').cpu().numpy())
        sft_like.append((head_value(hp, 'wp').cpu().numpy(), head_value(hn, 'wp').cpu().numpy()))
    ps, ns = np.concatenate(pos_s), np.concatenate(neg_s)
    cps, cns = np.concatenate([a for a, _ in sft_like]), np.concatenate([b for _, b in sft_like])
    lab = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
    out['uncond_AUC_watched_vs_popneg'] = {'RM-A uncond head': float(roc_auc_score(lab, np.r_[ps, ns])),
                                           'RM-A cond wp head': float(roc_auc_score(lab, np.r_[cps, cns]))}
    # return head on val-week rows whose 7-day window is observed
    Sv = rm_states(rm, [e['hist'] for e in ev_val]); rp, ry = [], []
    lim = np.datetime64('2021-08-14')
    for u, e in enumerate(ev_val):
        mk = (e['tgt'] >= 0) & (e['tdt'] <= lim)
        if not mk.any(): continue
        with torch.no_grad(): h = rm.head(Sv[u:u + 1], torch.from_numpy(e['tgt'][mk])[None].to(DEV))[0]
        rp.append(head_value(h, 'ret').cpu().numpy()); ry.append(e['tret'][mk])
    rp, ry = np.concatenate(rp), np.concatenate(ry)
    out['ret7_head_val'] = {'AUC': float(roc_auc_score(ry, rp)), 'base_rate': float(ry.mean())}
    results['rm_eval'] = out
    log('RM eval', json.dumps(out, indent=1))


def stage1():
    seed_all(0); results = {}
    train_seqs, ev, info, pop, stats = prepare(); results['data'] = stats
    train_mean_wp = float(np.concatenate([s[1] for s in train_seqs]).mean())
    sft = train_sft(train_seqs, ev['val'], info, results)
    torch.save(sft.state_dict(), f'{OUT}/sft.pt')
    rm = train_rm(sft, train_seqs, info, results)
    torch.save(rm.state_dict(), f'{OUT}/rm.pt')
    from catboost import CatBoostRegressor
    Xv, yv, gv = judge_pairs(ev['val'], info)
    tr_m = gv % 10 != 0
    judge = CatBoostRegressor(iterations=1500 if not SMOKE else 50, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=300)
    judge.fit(Xv[tr_m], yv[tr_m], eval_set=(Xv[~tr_m], yv[~tr_m]), early_stopping_rounds=100)
    judge.save_model(f'{OUT}/judge.cbm')
    eval_rm(rm, judge, ev['test'], ev['val'], info, train_mean_wp, results)
    jfn = make_judge_fn(judge, info)
    S = rm_states(rm, [e['hist'] for e in ev['test']]); ht = [e['hist'] for e in ev['test']]
    results['test'] = {}
    results['test']['Popularity'] = metrics(rec_pop(pop['test'], ht), ev['test'], info, rm, S, jfn)
    results['test']['SFT Top-K'] = metrics(recommend(sft, ht), ev['test'], info, rm, S, jfn)
    results['test']['SFT Top-K (no seen mask)'] = metrics(recommend(sft, ht, mask_seen=False), ev['test'], info, rm, S, jfn)
    results['test']['SFT Next-K (autoregressive)'] = metrics(next_k(sft, ht), ev['test'], info, rm, S, jfn)
    for k, v in results['test'].items(): log(k, {a: round(b, 4) for a, b in v.items()})
    json.dump(results, open(f'{OUT}/results_stage1.json', 'w'), indent=1)


def next_k(model, hists, k=K, bs=2048):
    """Autoregressive Next-K: append each generated item to the context (as if consumed)."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(hists), bs):
            cur = [h.copy() for h in hists[i:i + bs]]; gen = [[] for _ in cur]; x0, _ = pad(cur)
            for _ in range(k):
                x, m = pad([c[-MAXLEN:] for c in cur]); lg = policy_logits(model, x, m); lg.scatter_(1, x0, NEG)
                for r, g in enumerate(gen):
                    if g: lg[r, torch.tensor(g, device=DEV)] = NEG
                nx = lg.argmax(-1).cpu().numpy()
                for r in range(len(cur)): cur[r] = np.r_[cur[r], nx[r]]; gen[r].append(int(nx[r]))
            out.append(np.array(gen))
    return np.concatenate(out)


# ============================== STAGE 2 ==============================
def pl_logprob(logp, samp):
    """log-prob of ordered slates under Plackett-Luce (sampling w/o replacement). logp (B,V), samp (B,G,K)."""
    B, G, Kk = samp.shape
    lq = logp[:, None, :].expand(B, G, -1).gather(2, samp)
    q = lq.exp(); removed = torch.cumsum(q, -1) - q
    return (lq - torch.log((1 - removed).clamp_min(1e-6))).sum(-1)


def run_rl(sft, rm, states, info, name, beta, norm, steps=RL_STEPS, B=256, G=4, lr=3e-5, mask_seen=True, div=0.0, log_every=50):
    pol = make_gpt(info['V']); pol.load_state_dict(sft.state_dict()); pol.eval()
    ref = sft; ref.eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    w = (1 / torch.log2(torch.arange(2, K + 2, device=DEV).float())); w = w / w.sum()
    mu, sd = norm[name]; Gt = torch.tensor(info['G'], device=DEV)
    rs = np.random.RandomState(7); curve = []
    for step in range(steps):
        idx = rs.choice(len(states), B, replace=False)
        x, m = pad([states[i] for i in idx])
        lg = policy_logits(pol, x, m, mask_seen); logp = F.log_softmax(lg, -1)
        with torch.no_grad():
            lref = F.log_softmax(policy_logits(ref, x, m, mask_seen), -1)
            gum = -torch.log(-torch.log(torch.rand(B, G, logp.shape[1], device=DEV).clamp(1e-10, 1 - 1e-10)))
            samp = (logp.detach()[:, None, :] + gum).topk(K, -1).indices
            s = rm.state(x, m)
            r = (reward_value(rm, s[:, None, :], samp.view(B, -1), name).view(B, G, K) - mu) / sd
            R = (r * w).sum(-1)
            if div > 0:  # list-level reward: intra-list genre distance (non-decomposable)
                g = Gt[samp]; g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                sim = torch.einsum('bgkd,bgld->bgkl', g, g); ild = 1 - (sim.sum((-1, -2)) - K) / (K * (K - 1))
                R = R + div * ild
            A = R - (R.sum(1, keepdim=True) - R) / (G - 1)
            A = A / (A.std() + 1e-6)
        lp = pl_logprob(logp, samp)
        kl = (logp.exp() * (logp - lref)).sum(-1)
        loss = -(A * lp).mean() + beta * kl.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % log_every == 0 or step == steps - 1:
            ent = -(logp.exp() * logp).sum(-1).mean().item()
            curve.append({'step': step, 'R_sampled': R.mean().item(), 'KL': kl.mean().item(), 'entropy': ent})
            log(name, beta, curve[-1])
    pol.eval()
    return pol, curve


def rerank(sft, rm, hists, name, lam, norm, ncand=100, bs=2048):
    mu, sd = norm[name]; out = []
    with torch.no_grad():
        for i in range(0, len(hists), bs):
            x, m = pad(hists[i:i + bs]); lp = F.log_softmax(policy_logits(sft, x, m), -1)
            top = lp.topk(ncand, -1); s = rm.state(x, m)
            r = (reward_value(rm, s[:, None, :], top.indices, name) - mu) / sd
            sc = top.values + lam * r
            out.append(top.indices.gather(1, sc.topk(K, -1).indices).cpu().numpy())
    return np.concatenate(out)


def multistep(pol, rm, evs, info, steps=5, n=5000, bs=1000, name='wp'):
    """Generate `steps` items autoregressively, appending each to the context as if consumed."""
    evs = evs[:n]; hists = [e['hist'] for e in evs]
    hit_ar = np.zeros(steps); hit_os = np.zeros(steps); rm_real = np.zeros(steps); rm_syn = np.zeros(steps)
    kl_drift = np.zeros(steps); genre_ar, genre_os = [], []
    with torch.no_grad():
        for i in range(0, len(hists), bs):
            hb = hists[i:i + bs]; eb = evs[i:i + bs]
            x0, m0 = pad(hb); lg0 = policy_logits(pol, x0, m0); lp0 = F.log_softmax(lg0, -1)
            os_ = lg0.topk(steps, -1).indices.cpu().numpy(); s0 = rm.state(x0, m0)
            cur = [h.copy() for h in hb]; gen = [[] for _ in hb]
            for k in range(steps):
                x, m = pad([c[-MAXLEN:] for c in cur]); lg = policy_logits(pol, x, m); lg.scatter_(1, x0, NEG)
                for r, g in enumerate(gen):
                    if g: lg[r, torch.tensor(g, device=DEV)] = NEG
                lp = F.log_softmax(lg, -1); nx = lg.argmax(-1)
                # drift of the next-item distribution from the one conditioned on the real history
                kl_drift[k] += (lp.exp() * (lp - lp0)).sum(-1).sum().item()
                sk = rm.state(x, m)
                rm_syn[k] += reward_value(rm, sk[:, None, :], nx[:, None], name).sum().item()
                rm_real[k] += reward_value(rm, s0[:, None, :], nx[:, None], name).sum().item()
                nxn = nx.cpu().numpy()
                for r in range(len(cur)):
                    cur[r] = np.r_[cur[r], nxn[r]]; gen[r].append(int(nxn[r]))
                    v = eb[r]['tgt']; hit_ar[k] += nxn[r] in v; hit_os[k] += os_[r, k] in v
            for r in range(len(cur)):
                for arr, lst in [(np.array(gen[r]), genre_ar), (os_[r], genre_os)]:
                    Gm = info['G'][arr]; lst.append(Gm.sum(0).max() / max(Gm.sum(), 1))
    N = len(hists)
    return {'hit_autoregressive_step': (hit_ar / N).tolist(), 'hit_oneshot_rank': (hit_os / N).tolist(),
            f'RM_{name}_synthetic_ctx': (rm_syn / N).tolist(), f'RM_{name}_real_ctx': (rm_real / N).tolist(),
            'KL_drift_vs_real_history': (kl_drift / N).tolist(),
            'top_genre_share_autoregressive': float(np.mean(genre_ar)), 'top_genre_share_oneshot': float(np.mean(genre_os))}


def stage2():
    seed_all(0)
    train_seqs, ev, info, pop, stats = prepare()
    sft = make_gpt(info['V']); sft.load_state_dict(torch.load(find('sft.pt', PREV), map_location=DEV)); sft.eval()
    rm = RewardModel(sft, info['feat']).to(DEV); rm.load_state_dict(torch.load(find('rm.pt', PREV), map_location=DEV)); rm.eval()
    for p in rm.parameters(): p.requires_grad_(False)
    for p in sft.parameters(): p.requires_grad_(False)
    from catboost import CatBoostRegressor
    judge = CatBoostRegressor(); judge.load_model(find('judge.cbm', PREV)); jfn = make_judge_fn(judge, info)
    ht = [e['hist'] for e in ev['test']]; S = rm_states(rm, ht)
    states = [s[0][-MAXLEN:] for s in train_seqs]
    # reward normalisation: item-level stats of each reward over SFT top-10 on val users
    hv = [e['hist'] for e in ev['val']][:3000]; Sv = rm_states(rm, hv); rv = recommend(sft, hv)
    norm = {nm: (float(rm_scores(rm, Sv, rv, nm).mean()), float(rm_scores(rm, Sv, rv, nm).std() + 1e-6)) for nm in REWARDS}
    res = {'norm': norm, 'test': {}, 'rl_curves': {}, 'rerank': {}, 'multistep': {}}
    base = metrics(recommend(sft, ht), ev['test'], info, rm, S, jfn); res['test']['SFT'] = base
    pols = {}
    grid = [(nm, b) for nm in REWARDS for b in [0.0, 0.03, 0.1, 0.3, 1.0]]
    if SMOKE: grid = grid[:2]
    for nm, b in grid:
        pol, curve = run_rl(sft, rm, states, info, nm, b, norm)
        tag = f'RL[{nm}] beta={b}'
        res['test'][tag] = metrics(recommend(pol, ht), ev['test'], info, rm, S, jfn); res['rl_curves'][tag] = curve
        log(tag, {a: round(v, 4) for a, v in res['test'][tag].items()})
        if b in (0.1,): pols[nm] = pol
        json.dump(res, open(f'{OUT}/results_stage2.json', 'w'), indent=1)
    # ablations: no seen-mask (repeats), list-level diversity reward
    extra = [('RL[wp] beta=0.1 no-seen-mask', dict(name='wp', beta=0.1, mask_seen=False)),
             ('RL[uncond] beta=0.1 +ILD(1.0)', dict(name='uncond', beta=0.1, div=1.0))]
    for tag, kw in (extra if not SMOKE else extra[:1]):
        pol, curve = run_rl(sft, rm, states, info, kw.pop('name'), kw.pop('beta'), norm, **kw)
        ms = kw.get('mask_seen', True)
        res['test'][tag] = metrics(recommend(pol, ht, mask_seen=ms), ev['test'], info, rm, S, jfn); res['rl_curves'][tag] = curve
        log(tag, res['test'][tag])
    # re-ranking baseline: log pi_ref + lam * z(r) over top-100 SFT candidates (closed-form KL-regularised optimum, lam=1/beta)
    for nm in REWARDS:
        for lam in [0.3, 1, 3, 10, 30]:
            res['rerank'][f'rerank[{nm}] lam={lam}'] = metrics(rerank(sft, rm, ht, nm, lam, norm), ev['test'], info, rm, S, jfn)
    json.dump(res, open(f'{OUT}/results_stage2.json', 'w'), indent=1)
    # multi-step generation drift
    res['multistep']['SFT'] = multistep(sft, rm, ev['test'], info)
    for nm, pol in pols.items():
        res['multistep'][f'RL[{nm}] beta=0.1'] = multistep(pol, rm, ev['test'], info)
    json.dump(res, open(f'{OUT}/results_stage2.json', 'w'), indent=1)
    log('done')


def stage3():
    """Complete the beta sweep (strong KL) and check what happens without the step budget (long runs)."""
    seed_all(0)
    train_seqs, ev, info, pop, stats = prepare()
    sft = make_gpt(info['V']); sft.load_state_dict(torch.load(find('sft.pt', PREV), map_location=DEV)); sft.eval()
    rm = RewardModel(sft, info['feat']).to(DEV); rm.load_state_dict(torch.load(find('rm.pt', PREV), map_location=DEV)); rm.eval()
    for p in list(rm.parameters()) + list(sft.parameters()): p.requires_grad_(False)
    from catboost import CatBoostRegressor
    judge = CatBoostRegressor(); judge.load_model(find('judge.cbm', PREV)); jfn = make_judge_fn(judge, info)
    ht = [e['hist'] for e in ev['test']]; S = rm_states(rm, ht)
    states = [s[0][-MAXLEN:] for s in train_seqs]
    hv = [e['hist'] for e in ev['val']][:3000]; Sv = rm_states(rm, hv); rv = recommend(sft, hv)
    norm = {nm: (float(rm_scores(rm, Sv, rv, nm).mean()), float(rm_scores(rm, Sv, rv, nm).std() + 1e-6)) for nm in REWARDS}
    res = {'test': {}, 'rl_curves': {}}
    grid = [(nm, b, 400) for nm in REWARDS for b in [3.0, 10.0, 30.0]] +            [('wp', 0.0, 1500), ('wp', 1.0, 1500), ('wp', 10.0, 1500), ('uncond', 1.0, 1500), ('time', 1.0, 1500)]
    if SMOKE: grid = [('wp', 3.0, 4), ('wp', 0.0, 6)]
    for nm, b, st in grid:
        pol, curve = run_rl(sft, rm, states, info, nm, b, norm, steps=st, log_every=100)
        tag = f'RL[{nm}] beta={b}' + (f' steps={st}' if st != 400 else '')
        res['test'][tag] = metrics(recommend(pol, ht), ev['test'], info, rm, S, jfn); res['rl_curves'][tag] = curve
        log(tag, {a: round(v, 4) for a, v in res['test'][tag].items()})
        json.dump(res, open(f'{OUT}/results_stage3.json', 'w'), indent=1)
    log('done')


# ============================== STAGE 4 ==============================
def lpad(seqs):
    """Left padding, so that generated items can be appended on the right."""
    L = max(len(q) for q in seqs)
    x = np.zeros((len(seqs), L), np.int64); m = np.zeros((len(seqs), L), np.int64)
    for i, q in enumerate(seqs): x[i, L - len(q):] = q; m[i, L - len(q):] = 1
    return torch.from_numpy(x).to(DEV), torch.from_numpy(m).to(DEV)


def pos_ids(m):
    return (m.cumsum(1) - 1).clamp_min(0)


def run_rl_traj(sft, rm, states, info, name, beta, norm, steps=RL_STEPS, B=128, G=4, T=5, gamma=0.9, lr=3e-5, log_every=50):
    """Multi-step RL: the policy generates T items one by one, each appended to the context as if consumed.
    Step reward = RM prediction for the item given the (partly synthetic) context; reward-to-go with discount gamma;
    leave-one-out baseline over G trajectories of the same user; exact per-step KL to the SFT policy."""
    pol = make_gpt(info['V']); pol.load_state_dict(sft.state_dict()); pol.eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    mu, sd = norm[name]; V = info['V']
    disc = gamma ** torch.arange(T, device=DEV).float()
    rs = np.random.RandomState(11); curve = []
    for step in range(steps):
        idx = rs.choice(len(states), B, replace=False)
        H = [states[i][-(MAXLEN - T):] for i in idx for _ in range(G)]
        X, M = lpad(H); N = len(H)
        seen0 = torch.zeros(N, V, dtype=torch.bool, device=DEV); seen0.scatter_(1, X, True); seen0[:, :OFF] = True
        seen = seen0.clone(); gen = []
        with torch.no_grad():
            for t in range(T):
                h = pol.transformer(input_ids=X, attention_mask=M, position_ids=pos_ids(M)).last_hidden_state[:, -1]
                lg = pol.lm_head(h).masked_fill(seen, NEG)
                a = torch.multinomial(F.softmax(lg, -1), 1)
                X = torch.cat([X, a], 1); M = torch.cat([M, torch.ones_like(a)], 1); seen.scatter_(1, a, True); gen.append(a)
        Agen = torch.cat(gen, 1)  # (N, T)
        Lf = X.shape[1]; sl = slice(Lf - T - 1, Lf - 1)
        msk = seen0[:, None, :].repeat(1, T, 1)
        for t in range(1, T): msk[:, t].scatter_(1, Agen[:, :t], True)
        hid = pol.transformer(input_ids=X, attention_mask=M, position_ids=pos_ids(M)).last_hidden_state[:, sl]
        logp = F.log_softmax(pol.lm_head(hid).masked_fill(msk, NEG), -1)
        lpa = logp.gather(-1, Agen[..., None]).squeeze(-1)
        with torch.no_grad():
            hr = sft.transformer(input_ids=X, attention_mask=M, position_ids=pos_ids(M)).last_hidden_state[:, sl]
            lref = F.log_softmax(sft.lm_head(hr).masked_fill(msk, NEG), -1)
            Hs = rm.tr(input_ids=X, attention_mask=M, position_ids=pos_ids(M)).last_hidden_state[:, sl]
            r = (reward_value(rm, Hs, Agen, name) - mu) / sd  # (N, T)
            rtg = torch.stack([(r[:, t:] * disc[:T - t]).sum(-1) for t in range(T)], 1).view(B, G, T)
            A = rtg - (rtg.sum(1, keepdim=True) - rtg) / (G - 1)
            A = (A / (A.std() + 1e-6)).view(N, T)
        kl = (logp.exp() * (logp - lref)).sum(-1)  # (N, T)
        loss = -(A * lpa).sum(-1).mean() + beta * kl.sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append({'step': step, 'R_traj': (r * disc).sum(-1).mean().item(), 'KL': kl.sum(-1).mean().item()})
            log('traj', name, beta, curve[-1])
    pol.eval()
    return pol, curve


def run_offpolicy(sft, train_seqs, info, mode, steps=None, bs=256, lr=1e-4, gamma=0.9, horizon=5, n_factors=5, cap=1.0):
    """Training on logged interactions only, no reward model.
    mode='reinforce': off-policy REINFORCE, reward = real watched_pct (z-scored), return over the next `horizon`
      interactions of the user, importance weight = product of up to `n_factors` ratios pi_theta/pi_SFT, capped.
    mode='positive': next-item cross-entropy only on targets with watched_pct >= 80."""
    pol = make_gpt(info['V']); pol.load_state_dict(sft.state_dict()); pol.eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    allwp = np.concatenate([q[1] for q in train_seqs]); mu, sd = allwp.mean(), allwp.std()
    perm = np.random.RandomState(13).permutation(len(train_seqs))
    nsteps = steps or (len(perm) // bs); curve = []
    for st in range(nsteps):
        b = [train_seqs[j] for j in perm[st * bs:(st + 1) * bs]]
        x, m = pad([q[0] for q in b]); L = x.shape[1]
        wp = np.zeros((len(b), L), np.float32)
        for i, q in enumerate(b): wp[i, :len(q[1])] = q[1]
        wp = torch.from_numpy(wp).to(DEV)[:, 1:]; valid = m[:, 1:].float(); tgt = x[:, 1:]
        lp = F.log_softmax(pol(input_ids=x, attention_mask=m).logits[:, :-1], -1).gather(-1, tgt[..., None]).squeeze(-1)
        if mode == 'positive':
            pm = valid * (wp >= 80).float()
            loss = -(lp * pm).sum() / pm.sum().clamp_min(1)
        else:
            with torch.no_grad():
                lpb = F.log_softmax(sft(input_ids=x, attention_mask=m).logits[:, :-1], -1).gather(-1, tgt[..., None]).squeeze(-1)
                r = (wp - mu) / sd * valid
                R = sum((gamma ** j) * F.pad(r[:, j:], (0, j)) for j in range(horizon))
                lr_ = (lp.detach() - lpb) * valid
                logw = sum(F.pad(lr_[:, j:], (0, j)) for j in range(n_factors))
                w = logw.exp().clamp(max=cap)
                adv = (R - (R * valid).sum() / valid.sum()) * valid
            loss = -(w * adv * lp).sum() / valid.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if st % 200 == 0 or st == nsteps - 1:
            curve.append({'step': st, 'loss': loss.item()}); log('offpolicy', mode, curve[-1])
    pol.eval()
    return pol, curve


def eval_uncond(rm, evs, info, n=5000):
    """AUC of the unconditional head: items watched in the target week vs popularity-sampled / uniform negatives,
    plus the correlation of the head with item popularity on uniform items."""
    from sklearn.metrics import roc_auc_score
    evs = evs[:n]; S = rm_states(rm, [e['hist'] for e in evs]); rs = np.random.RandomState(3)
    popp = info['n_train'] / info['n_train'].sum(); P, NP, NU, IU = [], [], [], []
    for u, e in enumerate(evs):
        t = e['tgt'][e['tgt'] >= 0]
        if not len(t): continue
        nu = rs.randint(OFF, info['V'], len(t)); IU.append(nu)
        for arr, items in [(P, t), (NP, rs.choice(info['V'], len(t), p=popp)), (NU, nu)]:
            with torch.no_grad():
                arr.append(reward_value(rm, S[u:u + 1], torch.from_numpy(np.asarray(items))[None].to(DEV), 'uncond')[0].cpu().numpy())
    P, NP, NU, IU = map(np.concatenate, (P, NP, NU, IU))
    auc = lambda a, b: float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    return {'AUC_vs_popularity_neg': auc(P, NP), 'AUC_vs_uniform_neg': auc(P, NU),
            'corr_score_logpop_uniform_items': float(np.corrcoef(NU, np.log1p(info['n_train'][IU]))[0, 1])}


def stage4():
    seed_all(0)
    train_seqs, ev, info, pop, stats = prepare()
    sft = make_gpt(info['V']); sft.load_state_dict(torch.load(find('sft.pt', PREV), map_location=DEV)); sft.eval()
    rm = RewardModel(sft, info['feat']).to(DEV); rm.load_state_dict(torch.load(find('rm.pt', PREV), map_location=DEV)); rm.eval()
    attach_item_base(rm, info)
    for p in list(rm.parameters()) + list(sft.parameters()): p.requires_grad_(False)
    from catboost import CatBoostRegressor
    judge = CatBoostRegressor(); judge.load_model(find('judge.cbm', PREV)); jfn = make_judge_fn(judge, info)
    ht = [e['hist'] for e in ev['test']]; S = rm_states(rm, ht)
    states = [q[0][-MAXLEN:] for q in train_seqs]
    hv = [e['hist'] for e in ev['val']][:3000]; Sv = rm_states(rm, hv); rv = recommend(sft, hv)

    def make_norm(r_, Sv_, names):
        return {nm: (float(rm_scores(r_, Sv_, rv, nm).mean()), float(rm_scores(r_, Sv_, rv, nm).std() + 1e-6)) for nm in names}
    norm = make_norm(rm, Sv, REWARDS + ['wp_rel', 'time_rel'])
    res = {'norm': norm, 'test': {}, 'rl_curves': {}, 'multistep': {}, 'rm_variants': {}}
    save = lambda: json.dump(res, open(f'{OUT}/results_stage4.json', 'w'), indent=1)

    def add(tag, pol, curve=None):
        res['test'][tag] = metrics(recommend(pol, ht), ev['test'], info, rm, S, jfn)
        if curve is not None: res['rl_curves'][tag] = curve
        log(tag, {a: round(v, 4) for a, v in res['test'][tag].items()}); save()
    res['test']['SFT'] = metrics(recommend(sft, ht), ev['test'], info, rm, S, jfn); save()
    short = SMOKE > 0
    # A. item-relative rewards: engagement above the item's usual level
    grid = [('wp_rel', b, RL_STEPS) for b in [0.0, 1.0, 3.0]] + [('time_rel', b, RL_STEPS) for b in [0.0, 1.0, 3.0]] + \
           [('wp_rel', 0.0, 6 if short else 1500)]
    if short: grid = grid[:1] + grid[3:4] + grid[-1:]
    for nm, b, st in grid:
        pol, curve = run_rl(sft, rm, states, info, nm, b, norm, steps=st, log_every=100)
        add(f'RL[{nm}] beta={b}' + (f' steps={st}' if st != RL_STEPS else ''), pol, curve)
    for lam in [0.3, 1, 3]:
        res['test'][f'rerank[wp_rel] lam={lam}'] = metrics(rerank(sft, rm, ht, 'wp_rel', lam, norm), ev['test'], info, rm, S, jfn)
    save()
    # B. negative sampling of the unconditional head
    res['rm_variants']['pop'] = eval_uncond(rm, ev['test'], info); log('uncond pop', res['rm_variants']['pop'])
    for mode in ['uniform', 'mix']:
        rmv = train_rm(sft, train_seqs, info, {}, neg_mode=mode); attach_item_base(rmv, info)
        for p in rmv.parameters(): p.requires_grad_(False)
        res['rm_variants'][mode] = eval_uncond(rmv, ev['test'], info); log('uncond', mode, res['rm_variants'][mode])
        nv = make_norm(rmv, rm_states(rmv, hv), ['uncond']); Sx = rm_states(rmv, ht)
        for b in ([0.1, 1.0] if not short else [0.1]):
            pol, curve = run_rl(sft, rmv, states, info, 'uncond', b, nv, log_every=100)
            tag = f'RL[uncond, {mode} neg] beta={b}'; add(tag, pol, curve)
            res['test'][tag]['RMvariant_uncond'] = float(rm_scores(rmv, Sx, recommend(pol, ht), 'uncond').mean())
        save()
    # C. multi-step (trajectory) RL vs slate RL: does training on synthetic context exploit the RM?
    for nm, b in ([('wp', 1.0), ('wp', 0.1), ('time', 1.0)] if not short else [('wp', 1.0)]):
        pol, curve = run_rl_traj(sft, rm, states, info, nm, b, norm)
        tag = f'trajRL[{nm}] beta={b}'; add(tag, pol, curve)
        res['multistep'][tag] = multistep(pol, rm, ev['test'], info, name=nm)
        pol1, _ = run_rl(sft, rm, states, info, nm, b, norm, log_every=100)
        res['multistep'][f'RL[{nm}] beta={b}'] = multistep(pol1, rm, ev['test'], info, name=nm)
        save()
    res['multistep']['SFT (wp)'] = multistep(sft, rm, ev['test'], info, name='wp')
    res['multistep']['SFT (time)'] = multistep(sft, rm, ev['test'], info, name='time'); save()
    # D. training on logged feedback without a reward model
    for mode in ['reinforce', 'positive']:
        pol, curve = run_offpolicy(sft, train_seqs, info, mode, steps=4 if short else None)
        add(f'offpolicy[{mode}]', pol, curve)
    log('done')


if __name__ == '__main__':
    log('device', DEV, 'stage', STAGE, 'smoke', SMOKE)
    {'1': stage1, '2': stage2, '3': stage3, '4': stage4}[str(STAGE)]()
