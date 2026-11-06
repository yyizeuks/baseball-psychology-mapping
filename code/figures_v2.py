"""Figures for the revised manuscript (generation names fig1-fig7, figS1-figS5; manuscript numbering in export_tiff.py).
Reads out/*.json|csv produced by analysis_pipeline_v2.py, analysis_sensitivity_v2.py and derived_stats.py.
All text is drawn at >= 8 pt on canvases no wider than 180 mm (7.08 in), so that the exported figures meet the journal's
size and legibility requirements without rescaling."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.manifold import MDS
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

FS = 8  # minimum font size (pt)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': FS, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'axes.edgecolor': '#555', 'axes.labelcolor': '#222',
                     'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS, 'axes.labelsize': FS,
                     'axes.titlesize': FS, 'axes.unicode_minus': True})
W2 = 7.08   # double-column width (180 mm)
W1 = 3.35   # single-column width (85 mm)
OUT = 'out'; FIG = 'figs'
import os; os.makedirs(FIG, exist_ok=True)
L = json.load(open('labels.json'))
DOMS = sorted(L['domains'].keys())
T = {int(k): v for k, v in L['topics'].items()}; D = L['domains']
TC = {int(k): v for k, v in L['topic_colors'].items()}
ORDER = L['table_order']
diag = json.load(open(f'{OUT}/diagnostics.json'))
casc = diag['screening']
df = pd.read_csv(f'{OUT}/corpus_final.csv')
t1 = pd.read_csv(f'{OUT}/table1_topics.csv').set_index('topic_id')
t2 = pd.read_csv(f'{OUT}/table2_trends.csv').set_index('topic_id')
K = diag['K']
GREY = '#6b6b6b'

def mn(x, nd=1):
    """Signed number with a true minus sign."""
    s = f"{x:+.{nd}f}"
    if float(s) == 0 and x != 0: s = f"{x:+.{nd + 1}f}"  # a slope that rounds to zero keeps its sign at one more decimal
    elif float(s) == 0: s = f"{0:+.{nd}f}"
    return s.replace('-', '−')

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
    print('saved', name)

# ------------------------------------------------------------------ Figure 1: pipeline
def fig1():
    fig = plt.figure(figsize=(W2 - 0.08, 3.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.75); ax.axis('off')
    nd = len(DOMS)
    stages = [("1  Corpus construction", [f"3 sources, {casc['retrieved']:,} records", f"de-duplication (\u2212{casc['duplicates_removed']})",
                                          f"publication type (\u2212{casc['excluded_publication_type']})", "abstract, language,", "publication window",
                                          f"topicality rule (\u2212{casc['excluded_topicality']})", f"relevance screen \u2192 {casc['after_relevance_screen']}"]),
              ("2  Representation", ["sublinear TF-IDF", f"vocabulary {diag['vocabulary_size']:,} terms", "truncated SVD,", "120 components", "unit-normalized vectors"]),
              ("3  Topic induction", ["k-means, K = 6\u201318 sweep", "admissibility and", "coherence rule", f"K = {K} topics", f"{nd} data-derived domains", "c-TF-IDF signatures"]),
              ("4  Profile and dynamics", ["bibliometric profile", "share trends: OLS,", "GLM, CLR, Mann\u2013Kendall", "checks: K, dimension,", "seed, lexicon, source,", "MeSH indexing"])]
    cols = ['#0072B2', '#56B4E9', '#D55E00', '#009E73']
    for i, (title, lines) in enumerate(stages):
        x = 0.08 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, 0.12), 2.34, 3.5, boxstyle="round,pad=0.03,rounding_size=0.08", fc='white', ec=cols[i], lw=1.3))
        ax.add_patch(FancyBboxPatch((x, 3.12), 2.34, 0.5, boxstyle="round,pad=0.03,rounding_size=0.08", fc=cols[i], ec=cols[i], lw=1.3))
        ax.text(x + 1.17, 3.37, title, ha='center', va='center', fontsize=FS, fontweight='bold', color='white')
        for j, l in enumerate(lines):
            ax.text(x + 0.1, 2.78 - j * 0.4, l, fontsize=FS, va='center', color='#222')
        if i < 3: ax.add_patch(FancyArrowPatch((x + 2.36, 1.6), (x + 2.5, 1.6), arrowstyle='-|>', mutation_scale=9, color='#444', lw=1))
    save(fig, 'fig1_pipeline')

# ------------------------------------------------------------------ Figure 2: PRISMA-style flow
def fig2():
    fig = plt.figure(figsize=(W2 - 0.08, 5.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    def box(x, y, w, h, lines, fc='#FFFFFF', ec='#2C3E50', bold_first=False, fs=FS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.06", facecolor=fc, edgecolor=ec, linewidth=1.0))
        n = len(lines)
        for i, ln in enumerate(lines):
            yy = y + h - 0.27 - (i * (h - 0.34) / max(n - 0.5, 1)) if n > 1 else y + h / 2
            ax.text(x + 0.16, yy, ln, fontsize=fs, va='center', ha='left', fontweight='bold' if (bold_first and i == 0) else 'normal', color='#1a1a1a')
    def arrow(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, color='#444', linewidth=1.0))
    def stage(y, h, label, color):
        ax.add_patch(FancyBboxPatch((0.15, y), 0.6, h, boxstyle="round,pad=0.04,rounding_size=0.06", facecolor=color, edgecolor='none'))
        ax.text(0.45, y + h / 2, label, rotation=90, ha='center', va='center', fontsize=FS, fontweight='bold', color='white')
    stage(7.55, 2.3, 'Identification', '#2C6E9B'); stage(2.55, 4.6, 'Screening', '#4F9D69'); stage(0.1, 2.15, 'Included', '#C1553B')
    box(1.0, 7.75, 4.4, 2.0, ["Records identified from databases", f"(n = {casc['retrieved']:,})", "   Crossref (n = 362)", "   PubMed (n = 1,251)", "   Web of Science (n = 235)"], bold_first=True)
    box(5.9, 8.4, 4.0, 1.05, ["Records removed before screening", f"   Duplicate records (n = {casc['duplicates_removed']})"], bold_first=True)
    arrow(5.40, 8.9, 5.9, 8.9); arrow(3.2, 7.75, 3.2, 6.8)
    box(1.0, 5.85, 4.4, 0.95, ["Records screened on title, abstract,", f"and keywords (n = {casc['after_deduplication']:,})"], bold_first=True)
    nex = casc['excluded_publication_type'] + casc['excluded_abstract_total'] + casc['excluded_language'] + casc['excluded_year_window'] + casc['excluded_topicality'] + casc['excluded_relevance']
    box(5.9, 2.95, 4.0, 3.55, [f"Records excluded (n = {nex}):", f"   Non-research publication type (n = {casc['excluded_publication_type']})",
                                f"   Abstract absent or < 200", f"   characters (n = {casc['excluded_abstract_total']})",
                                f"   Non-English text (n = {casc['excluded_language']})", f"   Outside 1990–2026 window (n = {casc['excluded_year_window']})",
                                f"   Baseball not the object of study", f"   (n = {casc['excluded_topicality']})",
                                f"   Failed psychological-relevance", f"   screen (n = {casc['excluded_relevance']})"], bold_first=True)
    ax.plot([5.40, 5.85], [6.32, 6.32], color='#444', lw=1.0); arrow(5.7, 6.32, 5.9, 6.32)
    box(1.0, 3.2, 4.4, 1.55, ["Full-text retrieval not applicable:", "eligibility was assessed on the", "bibliographic record (title, abstract,", "keywords) for all records"], fc='#F4F6F8', ec='#8A97A5')
    arrow(3.2, 5.85, 3.2, 4.75); arrow(3.2, 3.2, 3.2, 2.25)
    box(1.0, 0.3, 4.4, 1.95, ["Records included in the mapping", f"synthesis (n = {casc['after_relevance_screen']})", f"   Topic modeling: n = {casc['after_relevance_screen']}",
                              "   Trend estimation (2005–2025):", f"   n = {diag['n_trend_window']} records"], bold_first=True)
    save(fig, 'fig2_prisma')

# ------------------------------------------------------------------ Figure 3: corpus growth (two panels, no dual axis)
def fig3():
    yrs = np.arange(1990, 2027); c = df.groupby('Publication Year').size().reindex(yrs, fill_value=0)
    fig, axes = plt.subplots(2, 1, figsize=(W2, 4.2), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
    axes[0].bar(yrs, c.values, color='#0072B2', width=0.8, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('Articles per year'); axes[0].axvspan(2025.5, 2026.5, color='#999', alpha=0.15)
    axes[0].text(2026, c.max() * 0.88, 'partial\nyear', fontsize=FS, ha='center', color=GREY)
    peak = c.idxmax(); axes[0].annotate(f'{peak}: {c.max()}', (peak, c.max()), xytext=(peak - 7, c.max() * 0.95), fontsize=FS, color='#222', arrowprops=dict(arrowstyle='-', color='#888', lw=0.6))
    axes[0].set_title('(A) Annual output', loc='left', fontsize=FS, fontweight='bold')
    axes[1].plot(yrs, c.cumsum().values, color='#D55E00', lw=1.8); axes[1].set_ylabel('Cumulative articles'); axes[1].set_xlabel('Publication year')
    for y in (2005, 2025): axes[1].axvline(y, color='#bbb', lw=0.6, ls=':')
    axes[1].text(2015, c.cumsum().max() * 0.12, 'trend window 2005–2025', fontsize=FS, ha='center', color=GREY)
    axes[1].set_title('(B) Cumulative output', loc='left', fontsize=FS, fontweight='bold')
    fig.tight_layout(); save(fig, 'fig3_corpus')

# ------------------------------------------------------------------ Figure 4: model selection
def fig4():
    sw = pd.DataFrame(diag['K_sweep']); rule = diag['K_rule']
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0))
    adm = sw.min_cluster >= rule['min_cluster']
    for ax, col, lab in zip(axes, ['umass', 'silhouette'], ['Mean $C_{\\mathrm{UMass}}$ coherence (higher is better)', 'Silhouette coefficient']):
        ax.plot(sw.K, sw[col], '-', color='#999', lw=1, zorder=1)
        ax.scatter(sw.K[adm], sw[col][adm], s=30, color='#0072B2', zorder=3, label='admissible (smallest topic ≥ 25 documents)')
        ax.scatter(sw.K[~adm], sw[col][~adm], s=30, facecolors='white', edgecolors='#0072B2', zorder=3, label='inadmissible')
        ax.axvline(K, color='#D55E00', lw=1, ls='--'); ax.set_xlabel('Number of topics (K)'); ax.set_ylabel(lab); ax.set_xticks(range(6, 19, 2))
    axes[0].annotate(f'selected K = {K}\n(highest admissible coherence)', (K, rule['best_umass']), xytext=(K + 2.2, rule['best_umass'] + 0.01), fontsize=FS, color='#D55E00', va='bottom',
                     arrowprops=dict(arrowstyle='-', color='#D55E00', lw=0.6))
    axes[1].set_ylim(sw.silhouette.min() - 0.003, sw.silhouette.max() + 0.014)
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=FS, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    axes[0].set_title('(A)', loc='left', fontsize=FS, fontweight='bold'); axes[1].set_title('(B)', loc='left', fontsize=FS, fontweight='bold')
    axes[0].set_ylim(sw.umass.min() - 0.03, rule['best_umass'] + 0.09)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig6_selection')

# ------------------------------------------------------------------ Figure 5: topic landscape
def fig5():
    cent = np.load(f'{OUT}/centroids.npy'); S = cent @ cent.T; Dm = 1 - S
    # classical MDS (principal coordinates) of the cosine distances: deterministic, and more faithful to the distance
    # matrix for seven points than the SMACOF solution (fidelity statistics written to out/landscape_fit.json)
    n_ = len(Dm); J_ = np.eye(n_) - np.ones((n_, n_)) / n_; B_ = -0.5 * J_ @ (Dm ** 2) @ J_
    w_, v_ = np.linalg.eigh(B_); idx_ = np.argsort(w_)[::-1]; w_, v_ = w_[idx_], v_[:, idx_]
    X = v_[:, :2] * np.sqrt(np.maximum(w_[:2], 0))
    d_ = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)); iu_ = np.triu_indices(n_, 1)
    from scipy.stats import spearmanr as _spr
    json.dump(dict(method='classical MDS (principal coordinates) of cosine distances between topic centroids',
                   variance_share_2d=round(float(w_[:2].sum() / w_[w_ > 0].sum()), 3),
                   stress1=round(float(np.sqrt(((d_[iu_] - Dm[iu_]) ** 2).sum() / (Dm[iu_] ** 2).sum())), 3),
                   spearman=round(float(_spr(d_[iu_], Dm[iu_]).correlation), 3)), open(f'{OUT}/landscape_fit.json', 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(W2 - 0.3, 4.8))
    # within-domain links make the data-derived grouping visible
    for d_name, dv in D.items():
        ts_ = dv['topics']
        for a_ in range(len(ts_)):
            for b_ in range(a_ + 1, len(ts_)):
                ax.plot([X[ts_[a_], 0], X[ts_[b_], 0]], [X[ts_[a_], 1], X[ts_[b_], 1]], color=dv['color'], lw=1.0, alpha=0.5, zorder=1)
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    placed = []  # display-space bboxes of markers and labels already placed
    for t in range(K):
        n = t1.loc[t, 'n']; col = D[T[t]['domain']]['color']
        ax.scatter(X[t, 0], X[t, 1], s=n * 3.0, color=col, alpha=0.75, edgecolor='white', linewidth=1.2, zorder=3)
    fig.canvas.draw()
    pad = 0.14; ax.set_xlim(X[:, 0].min() - pad * 3.6, X[:, 0].max() + pad * 3.6); ax.set_ylim(X[:, 1].min() - pad * 1.1, X[:, 1].max() + pad * 2.2)
    fig.canvas.draw()
    for t in range(K):
        n = t1.loc[t, 'n']; rad_pt = np.sqrt(n * 3.0 / np.pi) * 0.75
        cx, cy = ax.transData.transform((X[t, 0], X[t, 1])); r = rad_pt * fig.dpi / 72
        placed.append(Bbox([[cx - r, cy - r], [cx + r, cy + r]]))
    order = sorted(range(K), key=lambda t: -t1.loc[t, 'n'])
    axbox = ax.get_window_extent()
    for t in order:
        n = t1.loc[t, 'n']; rad = np.sqrt(n * 3.0 / np.pi) * 0.75
        lab = f"T{t}  {T[t]['short']}\n(n = {n})"
        dg = (rad + 4) * 0.8
        cands = [dict(xytext=(0, rad + 4), ha='center', va='bottom'), dict(xytext=(0, -rad - 4), ha='center', va='top'),
                 dict(xytext=(rad + 4, 0), ha='left', va='center'), dict(xytext=(-rad - 4, 0), ha='right', va='center'),
                 dict(xytext=(dg, dg), ha='left', va='bottom'), dict(xytext=(-dg, dg), ha='right', va='bottom'),
                 dict(xytext=(dg, -dg), ha='left', va='top'), dict(xytext=(-dg, -dg), ha='right', va='top'),
                 dict(xytext=(0, rad + 22), ha='center', va='bottom'), dict(xytext=(0, -rad - 22), ha='center', va='top'),
                 dict(xytext=(rad + 22, 0), ha='left', va='center'), dict(xytext=(-rad - 22, 0), ha='right', va='center')]
        best = None; best_pen = None
        for c in cands:
            a = ax.annotate(lab, (X[t, 0], X[t, 1]), textcoords='offset points', fontsize=FS, color='#222', zorder=4, **c)
            fig.canvas.draw(); bb = a.get_window_extent()
            inside_ax = bb.x0 >= axbox.x0 - 2 and bb.x1 <= axbox.x1 + 2 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1
            fb = fig.bbox; inside_fig = bb.x0 >= fb.x0 + 2 and bb.x1 <= fb.x1 - 2 and bb.y0 >= fb.y0 and bb.y1 <= fb.y1
            pen = sum(bb.overlaps(p) for i, p in enumerate(placed) if i != t) + (0 if inside_ax else (0.5 if inside_fig else 10))
            if best_pen is None or pen < best_pen: best, best_pen = (c, bb), pen
            a.remove()
        a = ax.annotate(lab, (X[t, 0], X[t, 1]), textcoords='offset points', fontsize=FS, color='#222', zorder=4, **best[0])
        fig.canvas.draw(); placed.append(a.get_window_extent())
    for d, v in D.items():
        ax.scatter([], [], color=v['color'], s=60, label=f"Domain {d}: {v['label']}")
    ax.legend(fontsize=FS, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False); ax.set_xticks([]); ax.set_yticks([])
    for sp in ('left', 'bottom'): ax.spines[sp].set_visible(False)
    ax.text(0.0, 1.0, 'Principal coordinates (classical MDS) of topic centroids, cosine distance; axes in arbitrary units\nmarker area ∝ documents; lines join the topics of a multi-topic domain', transform=ax.transAxes, fontsize=FS, va='bottom', color=GREY)
    save(fig, 'fig7_landscape')

# ------------------------------------------------------------------ Figure 6: trends forest with method agreement
def fig6():
    tm = json.load(open(f'{OUT}/sens_trendmeth.json'))['per_topic']
    yrs = np.arange(2005, 2026); sub = df[df['Publication Year'].isin(yrs)]
    tot = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0)
    fig, ax = plt.subplots(figsize=(W2, 0.42 * len(ORDER) + 1.9)); fig.subplots_adjust(left=0.40, right=0.98, top=0.97, bottom=0.30)
    ypos = np.arange(len(ORDER))[::-1]
    for y, t in zip(ypos, ORDER):
        cnt = sub[sub.topic == t].groupby('Publication Year').size().reindex(yrs, fill_value=0)
        prop = (cnt / tot.replace(0, np.nan)).fillna(0).values * 100
        sl, ic, r, p, se = stats.linregress(yrs, prop)
        b, ci = sl * 10, stats.t.ppf(0.975, len(yrs) - 2) * se * 10  # t-based 95% CI (df = 19)
        col = D[T[t]['domain']]['color']; r = tm[str(t)]
        nsig = sum(c in ('Hot', 'Cold') for c in (r['ols_cls'], r['glm_cls'], r['clr_cls'], r['mk_cls']))
        ax.plot([b - ci, b + ci], [y, y], color=col, lw=1.6, solid_capstyle='round')
        ax.scatter([b], [y], s=52, color=col if nsig >= 3 else 'white', edgecolor=col, linewidth=1.6, zorder=3)
        ax.text(b + ci + 0.6 if b >= 0 else b - ci - 0.6, y, f"{r['ols_cls'].split(' ')[0] if r['ols_cls'] in ('Hot', 'Cold') else 'n.s.'}  {nsig}/4", fontsize=FS, va='center', ha='left' if b >= 0 else 'right', color='#333')
    ax.axvline(0, ymax=0.92, color='#888', lw=0.8); ax.set_yticks(ypos); ax.set_yticklabels([f"T{t}  {T[t]['short']}" for t in ORDER], fontsize=FS)
    ax.set_xlabel('Change in annual topic share (percentage points per decade),\nOLS estimate with t-based 95% CI')
    ax.scatter([], [], s=52, color='#555', label='significant (adjusted p < 0.05) under ≥ 3 of 4 estimators'); ax.scatter([], [], s=52, facecolors='white', edgecolor='#555', label='fewer than 3 estimators')
    ax.legend(fontsize=FS, loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False, handletextpad=0.4)
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])) * 1.3; ax.set_xlim(-lim, lim)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.9)
    ax.text(0.99, 0.985, 'label: OLS class, estimators significant (OLS / GLM / CLR / Mann–Kendall); CIs unadjusted', transform=ax.transAxes, fontsize=FS, ha='right', va='top', color=GREY,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none'), zorder=4)
    save(fig, 'fig8_trends')

# ------------------------------------------------------------------ Figure 7: domains by era + annual share
def fig7():
    dt = json.load(open(f'{OUT}/domain_trends.json'))
    eras = [(1990, 1999), (2000, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.4), gridspec_kw={'width_ratios': [1, 1.3]})
    ax = axes[0]; bottom = np.zeros(len(eras)); labels = [f"{a}–{b}" for a, b in eras]
    for d in DOMS:
        vals = []
        for a, b in eras:
            m = df['Publication Year'].between(a, b); vals.append(100 * ((df.dom == d) & m).sum() / max(m.sum(), 1))
        ax.bar(range(len(eras)), vals, bottom=bottom, color=D[d]['color'], width=0.72, edgecolor='white', linewidth=1.2, label=f"{d}")
        for i, v in enumerate(vals):
            if v >= 4: ax.text(i, bottom[i] + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=FS, color='white')
        bottom += vals
    ax.set_xticks(range(len(eras))); ax.set_xticklabels(labels, fontsize=FS, rotation=25, ha='right'); ax.set_ylabel('Share of articles (%)')
    ax.set_title('(A) Domain composition by era', loc='left', fontsize=FS, fontweight='bold')
    ax = axes[1]; yrs = np.arange(2005, 2026); sub = df[df['Publication Year'].isin(yrs)]; tot = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0)
    ends = []
    for d in DOMS:
        cnt = sub[sub.dom == d].groupby('Publication Year').size().reindex(yrs, fill_value=0); prop = (cnt / tot).values * 100
        ax.plot(yrs, prop, 'o-', color=D[d]['color'], ms=3, lw=1.2, alpha=0.9)
        sl, ic, *_ = stats.linregress(yrs, prop); ax.plot(yrs, ic + sl * yrs, color=D[d]['color'], lw=1.6, ls='--')
        ends.append([d, ic + sl * 2025])
    ends.sort(key=lambda e: e[1])
    for k_ in range(1, len(ends)):  # push apart end labels closer than 6 points
        if ends[k_][1] - ends[k_ - 1][1] < 6: ends[k_][1] = ends[k_ - 1][1] + 6
    for d, yv in ends: ax.text(2025.5, yv, f"{d}  {mn(dt[d]['ols_b'])}/decade", fontsize=FS, va='center', color=D[d]['color'])
    ax.set_xlim(2004.5, 2030.5); ax.set_xticks([2005, 2010, 2015, 2020, 2025]); ax.set_ylabel('Annual share (%)'); ax.set_xlabel('Publication year')
    ax.set_title('(B) Annual domain shares and OLS fits', loc='left', fontsize=FS, fontweight='bold')
    handles = [plt.Line2D([], [], color=D[d]['color'], lw=3, label=f"{d}: {D[d]['label']}") for d in DOMS]
    fig.legend(handles=handles, fontsize=FS, loc='lower center', ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout(); save(fig, 'fig9_domains')

# ------------------------------------------------------------------ Supplementary figures
def figS1_krange():
    kr = json.load(open(f'{OUT}/sens_krange.json'))['per_K']
    Ks = sorted(int(k) for k in kr); fig, ax = plt.subplots(figsize=(W2, 0.36 * len(ORDER) + 2.2)); fig.subplots_adjust(left=0.33, right=0.99, top=0.92, bottom=0.30)
    cmap = {'Hot': '#D55E00', 'Cold': '#0072B2', 'positive (n.s.)': '#F4C7A5', 'negative (n.s.)': '#B8D4E8'}
    legend_names = {'Hot': 'Hot (positive, significant)', 'Cold': 'Cold (negative, significant)', 'positive (n.s.)': 'positive, n.s.', 'negative (n.s.)': 'negative, n.s.'}
    for i, t in enumerate(ORDER):
        for j, Kk in enumerate(Ks):
            m = kr[str(Kk)]['topics'][str(t)]
            weak = m['share_of_matched_from_primary'] < 0.5
            ax.add_patch(plt.Rectangle((j, len(ORDER) - 1 - i), 1, 1, facecolor=cmap[m['cls']], edgecolor='white', lw=1.5, hatch='////' if weak else None, alpha=0.55 if weak else 1))
            ax.text(j + 0.5, len(ORDER) - 1 - i + 0.5, mn(m['b_per_decade']), ha='center', va='center', fontsize=FS, color='white' if (m['cls'] in ('Hot', 'Cold') and not weak) else '#333')
    ax.set_xlim(0, len(Ks)); ax.set_ylim(0, len(ORDER)); ax.set_xticks(np.arange(len(Ks)) + 0.5); ax.set_xticklabels([f"K = {k}\nARI {kr[str(k)]['ari_vs_primary']:.2f}" for k in Ks], fontsize=FS)
    ax.set_yticks(np.arange(len(ORDER)) + 0.5); ax.set_yticklabels([f"T{t} {T[t]['short']}" for t in ORDER[::-1]], fontsize=FS)
    for sp in ('left', 'bottom'): ax.spines[sp].set_visible(False)
    for l, c in cmap.items(): ax.scatter([], [], marker='s', s=50, color=c, label=legend_names[l])
    ax.scatter([], [], marker='s', s=50, facecolor='#ddd', edgecolor='#999', hatch='////', label='match shares < 50% of documents with the primary topic')
    ax.legend(fontsize=FS, ncol=2, loc='upper center', bbox_to_anchor=(0.35, -0.3), frameon=False, columnspacing=1.0)
    ax.set_title('Trend class of the best-matching topic across K\n(cell = OLS slope, percentage points per decade)', fontsize=FS, loc='left')
    save(fig, 'figS7_krange')

def figS2_domains_dendro():
    sd = json.load(open(f'{OUT}/sens_domains.json')); Z = np.array(sd['linkage']); co = np.array(sd['bootstrap_cogrouping'])
    fig = plt.figure(figsize=(W2 - 1.0, 6.4))
    ax0 = fig.add_axes([0.42, 0.60, 0.55, 0.34]); ax1 = fig.add_axes([0.42, 0.06, 0.40, 0.42])
    dendrogram(Z, labels=[f"T{t} {T[t]['short']}" for t in range(K)], ax=ax0, orientation='right', color_threshold=0, above_threshold_color='#555')
    ax0.set_xlabel('Cosine distance (average linkage)'); ax0.tick_params(axis='y', labelsize=FS)
    ax0.set_title('(A) Hierarchical clustering of topic centroids', loc='left', fontsize=FS, fontweight='bold')
    im = ax1.imshow(co, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(K)); ax1.set_yticks(range(K)); ax1.set_xticklabels([f'T{t}' for t in range(K)], fontsize=FS); ax1.set_yticklabels([f'T{t} {T[t]['short']}' for t in range(K)], fontsize=FS)
    for i in range(K):
        for j in range(K):
            if i != j and co[i, j] >= 0.5: ax1.text(j, i, f'{100 * co[i, j]:.1f}', ha='center', va='center', fontsize=FS, color='white' if co[i, j] > 0.6 else '#222')
    ax1.set_title(f"(B) Bootstrap co-grouping at the {sd['n_domains']}-domain cut (% of 200 resamples)", loc='left', fontsize=FS, fontweight='bold')
    cax = fig.add_axes([0.84, 0.10, 0.02, 0.34]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0]); cb.set_ticklabels(['0', '25', '50', '75', '100']); cb.set_label('% of resamples', fontsize=FS)
    save(fig, 'figS5_domains')

def figS3_bootstrap_lexicon():
    bs = json.load(open(f'{OUT}/sens_bootstrap.json')); lx = json.load(open(f'{OUT}/sens_lexicon.json')); lx2 = json.load(open(f'{OUT}/sens_lexicon2.json'))
    fig, axes = plt.subplots(2, 1, figsize=(W2 - 0.6, 6.4), gridspec_kw={'height_ratios': [1, 1.15], 'hspace': 0.55})
    ax = axes[0]; vals = [bs['mean_jaccard'][t] for t in ORDER]; cols = [D[T[t]['domain']]['color'] for t in ORDER]
    ax.barh(range(len(ORDER))[::-1], vals, color=cols, height=0.65)
    for xv, c in ((0.75, '#444'), (0.6, '#777'), (0.5, '#aaa')): ax.axvline(xv, color=c, ls='--' if xv == 0.75 else ':', lw=0.8)
    ax.text(0.0, len(ORDER) - 0.3, 'Hennig (2007): \u2265 0.75 stable (dashed line);\n0.60\u20130.75 recognizable pattern; 0.50\u20130.60 not stable;\n< 0.50 dissolved (dotted lines at 0.50 and 0.60)', fontsize=FS, color='#555', va='bottom', ha='left', zorder=5, bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.set_yticks(range(len(ORDER))[::-1]); ax.set_yticklabels([f"T{t} {T[t]['short']}" for t in ORDER], fontsize=FS); ax.set_xlabel(f"Mean bootstrap Jaccard ({bs['n_boot']} resamples)", fontsize=FS); ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(ORDER) + 2.4); ax.set_title('(A) Topic-level bootstrap stability', loc='left', fontsize=FS, fontweight='bold')
    ax = axes[1]
    def matched(lx_):  # slope of the matched refitted topic; blank where the match shares < 50% of its documents with the primary topic
        out = []
        for t in ORDER:
            x = next((x for x in lx_['topics'] if x['primary_match'] == t), None)
            out.append(x['b'] if (x is not None and x.get('match_quality', 1) >= 0.5) else np.nan)
        return out
    rows = [('Primary lexicon', len(df), [t2.loc[t, 'b_per_decade'] for t in ORDER]),
            ('Strict lexicon', lx['n_strict'], matched(lx)),
            ("Strict, 'stress' removed", lx2['n'], matched(lx2))]
    w = 0.26
    for i, (name, n, b) in enumerate(rows):
        ax.barh(np.arange(len(ORDER))[::-1] + (1 - i) * w, np.nan_to_num(b), height=w, color=['#0072B2', '#56B4E9', '#B8D4E8'][i], label=f'{name} (n = {n})')
    for j, t in enumerate(ORDER):
        y0 = np.arange(len(ORDER))[::-1][j]; miss = [np.isnan(rows[i][2][j]) for i in (1, 2)]
        if all(miss): ax.text(0.3, y0 - w / 2, 'no counterpart (both strict lexicons)', fontsize=FS, va='center', color='#999')
        elif miss[0]: ax.text(0.3, y0, 'no counterpart (strict lexicon)', fontsize=FS, va='center', color='#999')
        elif miss[1]: ax.text(0.3, y0 - w, "no counterpart (strict, 'stress' removed)", fontsize=FS, va='center', color='#999')
    ax.axvline(0, color='#888', lw=0.8); ax.set_yticks(range(len(ORDER))[::-1]); ax.set_yticklabels([f"T{t}" for t in ORDER], fontsize=FS); ax.set_xlabel('OLS slope of matched topic (percentage points per decade)', fontsize=FS)
    xl = ax.get_xlim(); ax.set_xlim(xl[0], max(xl[1], 0) + 0.95 * (xl[1] - xl[0]))
    ax.legend(fontsize=FS, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1); ax.set_title('(B) Lexicon sensitivity', loc='left', fontsize=FS, fontweight='bold')
    ax.set_yticklabels([f"T{t} {T[t]['short']}" for t in ORDER], fontsize=FS)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.95, bottom=0.17); save(fig, 'figS4_stability_lexicon')

def figS4_abstract_period():
    ab = casc['excluded_abstract_by_period']; tot = casc['records_by_period_after_dedup']
    per = list(ab.keys()); ex = np.array([ab[p] for p in per]); to = np.array([tot[p] for p in per])
    fig, ax = plt.subplots(figsize=(5.0, 2.8)); x = np.arange(len(per))
    ax.bar(x, 100 * ex / np.maximum(to, 1), color='#0072B2', width=0.6)
    for i in range(len(per)): ax.text(i, 100 * ex[i] / max(to[i], 1) + 1, f'{ex[i]}/{to[i]}', ha='center', fontsize=FS)
    ax.set_xticks(x); ax.set_xticklabels([p.replace('-', '–') if p[0].isdigit() else p for p in per], fontsize=FS); ax.set_xlabel('Publication period'); ax.set_ylabel('Records excluded for abstract\nabsence or length (%)'); ax.set_ylim(0, max(100 * ex / np.maximum(to, 1)) * 1.3)
    save(fig, 'figS1_abstract_period')

def figS5_substructure():
    ss = json.load(open(f'{OUT}/sens_substruct.json'))
    fig, axes = plt.subplots(len(ss), 1, figsize=(W2 - 0.7, 1.75 * len(ss) + 0.3))
    for ax, (t, v) in zip(np.atleast_1d(axes), ss.items()):
        subs = v['splits']['3']['sub']; names = [', '.join(s['kw'][:4]) for s in subs]; ns = [s['n'] for s in subs]
        ax.barh(range(len(ns))[::-1], ns, color=D[T[int(t)]['domain']]['color'], height=0.6)
        ax.set_yticks(range(len(ns))[::-1]); ax.set_yticklabels(names, fontsize=FS); ax.set_xlabel('Documents')
        ax.set_title(f"T{t} {T[int(t)]['short']} (n = {v['n']}): sub-structure at k = 3", fontsize=FS, loc='left', fontweight='bold')
    fig.tight_layout(); save(fig, 'figS6_substructure')

if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, figS1_krange, figS2_domains_dendro, figS3_bootstrap_lexicon, figS4_abstract_period, figS5_substructure):
        try: f()
        except Exception as e: print('FAILED', f.__name__, type(e).__name__, e)
