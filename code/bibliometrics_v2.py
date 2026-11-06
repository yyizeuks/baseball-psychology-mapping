"""Bibliometric profile of the screened corpus (Reviewer 2, Section 2.4 request).

Inputs : out/corpus_final.csv (topics, domains), out/enriched_pubmed.csv (PubMed metadata)
Outputs: out/biblio.json (all numbers), out/table_top_cited.csv, out/table_productivity.csv,
         figs/fig5_coword.png, figs/fig4_countries.png, figs/figS3_mesh.png
Citation counts are PubMed/PMC 'cited-in' links: citations from PubMed-indexed articles only.
"""
import json, re, itertools, collections
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'legend.fontsize': 8, 'axes.titlesize': 8})

OUT, FIG = 'out', 'figs'
L = json.load(open('labels.json')); T = {int(k): v for k, v in L['topics'].items()}; D = L['domains']
DOMS = sorted(D.keys())
df = pd.read_csv(f'{OUT}/corpus_final.csv'); en = pd.read_csv(f'{OUT}/enriched_pubmed.csv')
en = en.sort_values('corpus_index').reset_index(drop=True)
assert len(en) == len(df)
en['topic'] = df['topic'].values; en['dom'] = df['dom'].values; en['year'] = df['Publication Year'].values
m = en['pmid'].notna() & (en['pmid'].astype(str) != '') & (en['pmid'].astype(str) != 'nan')
en['matched'] = m
B = {'n': int(len(en)), 'matched': int(m.sum()), 'match_rate': round(float(m.mean()), 3),
     'match_by_domain': {d: round(float(en[en.dom == d]['matched'].mean()), 3) for d in DOMS},
     'match_by_period': {p: round(float(en[en.year.between(a, b)]['matched'].mean()), 3) for a, b, p in [(1990, 2004, '1990-2004'), (2005, 2014, '2005-2014'), (2015, 2026, '2015-2026')]}}

# ---------------------------------------------------------------- countries
COUNTRIES = {'United States': ['usa', 'united states', 'u.s.a', ' us ', 'u.s.'], 'United Kingdom': ['united kingdom', ' uk', 'england', 'scotland', 'wales'],
             'Japan': ['japan'], 'South Korea': ['korea'], 'Taiwan': ['taiwan'], 'China': ['china', 'hong kong'], 'Australia': ['australia'], 'Canada': ['canada'],
             'Germany': ['germany'], 'Netherlands': ['netherlands'], 'Brazil': ['brazil', 'brasil'], 'Spain': ['spain'], 'Italy': ['italy'], 'France': ['france'],
             'Sweden': ['sweden'], 'Belgium': ['belgium'], 'Switzerland': ['switzerland'], 'Denmark': ['denmark'], 'Norway': ['norway'], 'Finland': ['finland'],
             'New Zealand': ['new zealand'], 'Israel': ['israel'], 'India': ['india'], 'Mexico': ['mexico'], 'Dominican Republic': ['dominican'], 'Cuba': ['cuba'],
             'Venezuela': ['venezuela'], 'Puerto Rico': ['puerto rico'], 'Singapore': ['singapore'], 'Ireland': ['ireland'], 'Poland': ['poland'], 'Portugal': ['portugal'],
             'Austria': ['austria'], 'Greece': ['greece'], 'Turkey': ['turkey', 'türkiye'], 'Iran': ['iran'], 'Saudi Arabia': ['saudi'], 'Thailand': ['thailand'],
             'Malaysia': ['malaysia'], 'Indonesia': ['indonesia'], 'Philippines': ['philippines'], 'Argentina': ['argentina'], 'Chile': ['chile'], 'Colombia': ['colombia'],
             'Czech Republic': ['czech'], 'Hungary': ['hungary'], 'Russia': ['russia'], 'South Africa': ['south africa'], 'Nigeria': ['nigeria'], 'Egypt': ['egypt'],
             'Qatar': ['qatar'], 'Nicaragua': ['nicaragua'], 'Panama': ['panama'], 'Slovenia': ['slovenia'], 'Croatia': ['croatia'], 'Serbia': ['serbia']}
US_STATE = re.compile(r',\s*(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b')
def countries_of(aff):
    found = set()
    for seg in str(aff).split('||'):
        s = ' ' + seg.lower() + ' '
        for c, keys in COUNTRIES.items():
            if any(re.search(r'(?<![a-z])' + re.escape(k.strip()) + r'(?![a-z])', s) for k in keys): found.add(c)
        if US_STATE.search(seg) or re.search(r'\b\d{5}(-\d{4})?\b', seg) and 'usa' not in s and not found: found.add('United States')
    return found
en['countries'] = en['affiliations'].apply(lambda a: countries_of(a) if isinstance(a, str) and a.strip() else set())
has_c = en['countries'].apply(len) > 0
B['country_coverage'] = {'records_with_affiliation': int((en['affiliations'].fillna('').str.len() > 0).sum()), 'records_with_country': int(has_c.sum())}
cc = collections.Counter(c for s in en['countries'] for c in s)
B['top_countries'] = cc.most_common(15)
intl = en[has_c]['countries'].apply(lambda s: len(s) > 1)
B['international_collaboration_rate'] = round(float(intl.mean()), 3)
# country network
G = nx.Graph()
for s in en['countries']:
    for c in s: G.add_node(c, w=G.nodes[c]['w'] + 1 if c in G else 1)
    for a, b in itertools.combinations(sorted(s), 2):
        G.add_edge(a, b, w=G[a][b]['w'] + 1 if G.has_edge(a, b) else 1)
B['country_edges_top'] = sorted([(a, b, d['w']) for a, b, d in G.edges(data=True)], key=lambda x: -x[2])[:12]
# countries by domain
B['country_by_domain'] = {d: collections.Counter(c for s in en[en.dom == d]['countries'] for c in s).most_common(6) for d in DOMS}
# East-Asian share
ea = {'Japan', 'South Korea', 'Taiwan', 'China'}
B['east_asia_share_of_country_records'] = round(float(en[has_c]['countries'].apply(lambda s: bool(s & ea)).mean()), 3)

# ---------------------------------------------------------------- institutions (first author, affiliation string normalized)
ORG_STRONG = re.compile(r'universit|college|hospital|academy|universidad|universit[aä]t', re.I)
ORG_MID = re.compile(r'medical cent(er|re)|institute|clinic|foundation|health system|infirmary|polytechnic', re.I)
ORG_WEAK = re.compile(r'cent(er|re)|school of|laborator|research', re.I)
SUBUNIT = re.compile(r'^(department|dept|division|school of|faculty of|center for|centre for|research cent|laboratory|lab |unit |program|section|graduate school|institute of (sport|kinesiology|exercise|psychology))', re.I)
def inst_of(aff):
    """Organization-level institution of one affiliation string (first affiliation of the first author if several)."""
    if not isinstance(aff, str) or not aff.strip(): return None
    first = aff.split('||')[0].split(';')[0]
    segs = [re.sub(r'^(the )', '', x.strip(), flags=re.I) for x in first.split(',')]
    best, best_score = None, -9
    for k, seg in enumerate(segs):
        if not seg or len(seg) >= 80 or re.search(r'@|\d{4,}', seg): continue
        if ORG_STRONG.search(seg): sc = 3
        elif ORG_MID.search(seg): sc = 2
        elif ORG_WEAK.search(seg): sc = 1
        else: continue
        if SUBUNIT.search(seg): sc -= 2
        if sc >= best_score: best, best_score = seg, sc   # ties -> later segment (organization usually follows the sub-unit)
    if best is None: return None
    if best in ('University of California', 'University of Texas', 'State University of New York', 'University of Wisconsin', 'University of Illinois', 'University of Colorado'):
        i = segs.index(best) if best in segs else -1
        if 0 <= i < len(segs) - 1 and len(segs[i + 1]) < 30 and not re.search(r'\d|department|school|division', segs[i + 1], re.I):
            best = best + ', ' + segs[i + 1]
    return best
en['inst'] = en['affiliations'].apply(inst_of)
B['top_institutions'] = collections.Counter(en['inst'].dropna()).most_common(15)
# inter-institutional co-authorship: records with affiliation strings for at least two authors
def insts_all(aff):
    if not isinstance(aff, str) or not aff.strip(): return [], 0
    segs = [x for x in aff.split('||') if x.strip()]
    found = {re.sub(r'[\s\.;:]+$', '', inst_of(x).lower()) for x in segs if inst_of(x)}
    return sorted(found), len(segs)
_ia = en['affiliations'].apply(insts_all)
en['insts'] = _ia.apply(lambda t: t[0]); en['n_aff_segments'] = _ia.apply(lambda t: t[1])
_multi = en[(en['n_aff_segments'] >= 2) & (en['insts'].apply(len) >= 1)]
B['institution_collaboration'] = dict(records_with_multi_author_affiliations=int(len(_multi)),
                                      multi_institution_rate=round(float((_multi['insts'].apply(len) >= 2).mean()), 3),
                                      mean_institutions=round(float(_multi['insts'].apply(len).mean()), 2))

# ---------------------------------------------------------------- journals
SMALL = {'of', 'and', 'the', 'in', 'for', 'on', 'to', 'a', 'an', 'de', 'et', 'la', 'le', 'des', 'du'}
SPECIAL = {'plos one': 'PLOS ONE', 'jses international': 'JSES International', 'jses reviews, reports, and techniques': 'JSES Reviews, Reports, and Techniques',
           'bmj open sport & exercise medicine': 'BMJ Open Sport & Exercise Medicine', 'hss journal : the musculoskeletal journal of hospital for special surgery': 'HSS Journal',
           'peerj': 'PeerJ', 'the journal of the american academy of orthopaedic surgeons': 'Journal of the American Academy of Orthopaedic Surgeons'}
def tcase(j):
    j = re.sub(r'\s+', ' ', str(j).strip()); lj = j.lower()
    if lj in SPECIAL: return SPECIAL[lj]
    lj = re.sub(r'^the ', '', lj); lj = lj.split(' : ')[0].split(' = ')[0]
    words = lj.split(' ')
    out = [w if (w in SMALL and i > 0) else (w.upper() if w in ('ieee', 'bmc', 'bmj', 'jama', 'mri') else w.capitalize()) for i, w in enumerate(words)]
    return ' '.join(out)
en['journal_c'] = en['journal'].fillna('').apply(lambda j: tcase(j) if str(j).strip() else '')
jc = collections.Counter(j for j in en['journal_c'] if j)
B['n_journals'] = len(jc); B['top_journals'] = jc.most_common(20)
B['top_journals_by_domain'] = {d: collections.Counter(j for j in en[en.dom == d]['journal_c'] if j).most_common(5) for d in DOMS}
B['top_journals_by_topic'] = {t: collections.Counter(j for j in en[en.topic == t]['journal_c'] if j).most_common(3) for t in range(len(T))}
# venue overlap between the growth areas and between domains (Jaccard of journal sets)
def jset(mask): return set(j for j in en[mask]['journal_c'] if j)
def jacc(a, b): return round(len(a & b) / max(len(a | b), 1), 3)
B['journal_overlap'] = {'T5_vs_T4': jacc(jset(en.topic == 5), jset(en.topic == 4)), 'T0_vs_T4': jacc(jset(en.topic == 0), jset(en.topic == 4)),
                        'T1_vs_T4': jacc(jset(en.topic == 1), jset(en.topic == 4)), 'T5_vs_T6': jacc(jset(en.topic == 5), jset(en.topic == 6)),
                        'A_vs_B': jacc(jset(en.dom == 'A'), jset(en.dom == 'B')), 'A_vs_C': jacc(jset(en.dom == 'A'), jset(en.dom == 'C')), 'B_vs_C': jacc(jset(en.dom == 'B'), jset(en.dom == 'C'))}
# author overlap between domains (share of authors with papers in both)
def norm_author(a):
    a = a.strip()
    if ',' not in a: return a
    last, first = [x.strip() for x in a.split(',', 1)]
    ini = ''.join(w[0] for w in re.split(r'[\s\-]+', first) if w)[:2]
    return f"{last} {ini}"
auth = {d: set(norm_author(a) for s in en[en.dom == d]['authors'].dropna() for a in str(s).split('|') if a.strip()) for d in DOMS}
B['author_overlap'] = {f'{a}_vs_{b}': dict(shared=len(auth[a] & auth[b]), jaccard=jacc(auth[a], auth[b])) for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]}
B['n_unique_authors_matched'] = len(set().union(*auth.values()))
en['n_authors'] = pd.to_numeric(en['n_authors'], errors='coerce')
B['mean_authors_per_paper'] = round(float(en['n_authors'].mean()), 2)
B['single_author_share'] = round(float((en['n_authors'] == 1).mean()), 3)

# ---------------------------------------------------------------- citations (PubMed/PMC cited-in)
en['cites'] = pd.to_numeric(en['pubmed_citations'], errors='coerce')
c = en[en.matched]['cites'].fillna(0).astype(int).values
cs = np.sort(c)[::-1]; h = int(np.sum(cs >= np.arange(1, len(cs) + 1)))
B['citations'] = {'basis': 'PubMed/PMC cited-in links', 'n_with_counts': int(en.matched.sum()), 'total': int(c.sum()), 'mean': round(float(c.mean()), 2),
                  'median': float(np.median(c)), 'h_index': h, 'uncited_share': round(float((c == 0).mean()), 3), 'max': int(c.max())}
en['cpy'] = en['cites'] / (2026 - en['year'] + 1)
B['citations_by_domain'] = {d: dict(mean=round(float(en[(en.dom == d) & en.matched]['cites'].mean()), 2), median=float(en[(en.dom == d) & en.matched]['cites'].median()),
                                    mean_per_year=round(float(en[(en.dom == d) & en.matched]['cpy'].mean()), 2)) for d in DOMS}
B['citations_by_topic'] = {t: dict(mean=round(float(en[(en.topic == t) & en.matched]['cites'].mean()), 2), median=float(en[(en.topic == t) & en.matched]['cites'].median()),
                                   mean_per_year=round(float(en[(en.topic == t) & en.matched]['cpy'].mean()), 2)) for t in range(len(T))}
top = en[en.matched].sort_values('cites', ascending=False).head(20)
pd.DataFrame(dict(rank=range(1, len(top) + 1), title=top['title'].values, year=top['year'].values, journal=top['journal_abbrev'].values,
                  first_author=top['authors'].fillna('').apply(lambda s: s.split('|')[0].strip()).values, topic=top['topic'].values,
                  domain=top['dom'].values, citations=top['cites'].astype(int).values,
                  cites_per_year=(np.floor(top['cpy'].values * 10 + 0.5) / 10))).to_csv(f'{OUT}/table_top_cited.csv', index=False)
B['top_cited_domains'] = collections.Counter(top['dom']).most_common()

# ---------------------------------------------------------------- keyword co-occurrence (author keywords)
STOPKW = {'baseball', 'sport', 'sports', 'athletes', 'athlete', 'humans', 'human', 'male', 'female', 'adult', 'young adult', 'adolescent', 'child',
          'middle aged', 'aged', 'animals', 'animal', 'united states', 'cross-sectional studies', 'surveys and questionnaires', 'retrospective studies',
          'prospective studies', 'follow-up studies', 'cohort studies', 'case-control studies', 'questionnaires', 'aged, 80 and over', 'child, preschool',
          'infant', 'time factors', 'pilot projects', 'reproducibility of results', 'analysis of variance', 'regression analysis', 'logistic models',
          'statistics as topic', 'data interpretation, statistical', 'students', 'universities', 'japan', 'republic of korea', 'taiwan', 'korea'}
def norm_kw(k):
    k = re.sub(r'\s+', ' ', k.strip().lower())
    return k
docs_kw = []
for k, t, d in zip(df['Keywords'], df['topic'], df['dom']):
    if isinstance(k, str) and k.strip():
        s = {norm_kw(x) for x in k.split(';') if 2 < len(x.strip()) < 60}
        s = {x for x in s if x and x not in STOPKW}
        docs_kw.append((s, t, d))
    else: docs_kw.append((set(), t, d))
B['keyword_coverage'] = {'records_with_keywords': int(sum(1 for s, _, _ in docs_kw if s)), 'share': round(float(np.mean([bool(s) for s, _, _ in docs_kw])), 3)}
kc0 = collections.Counter(x for s, _, _ in docs_kw for x in s)
def canon(x):
    if x.endswith('s') and not x.endswith('ss') and x[:-1] in kc0: return x[:-1]
    if x.endswith('ies') and x[:-3] + 'y' in kc0: return x[:-3] + 'y'
    return x
docs_kw = [({canon(x) for x in s}, t, d) for s, t, d in docs_kw]
kc = collections.Counter(x for s, _, _ in docs_kw for x in s)
vocab = [k for k, v in kc.items() if v >= 5]
B['n_keywords_total'] = len(kc); B['n_keywords_min5'] = len(vocab); B['top_keywords'] = kc.most_common(30)
Gk = nx.Graph()
for s, _, _ in docs_kw:
    ss = sorted(x for x in s if x in set(vocab))
    for x in ss: Gk.add_node(x, w=kc[x])
    for a, b in itertools.combinations(ss, 2): Gk.add_edge(a, b, w=Gk[a][b]['w'] + 1 if Gk.has_edge(a, b) else 1)
Gk.remove_edges_from([(a, b) for a, b, d in Gk.edges(data=True) if d['w'] < 2]); Gk.remove_nodes_from(list(nx.isolates(Gk)))
comms = nx.community.louvain_communities(Gk, weight='w', seed=42, resolution=1.0)
comms = sorted(comms, key=len, reverse=True)
kw2c = {k: i for i, c in enumerate(comms) for k in c}
# community-topic cross-tab: for each doc, fraction of its keywords in each community
ct = np.zeros((len(comms), len(T)))
for s, t, d in docs_kw:
    cs_ = [kw2c[x] for x in s if x in kw2c]
    for ci in cs_: ct[ci, t] += 1 / len(cs_)
B['coword'] = {'n_nodes': Gk.number_of_nodes(), 'n_edges': Gk.number_of_edges(), 'n_communities': len(comms),
               'modularity': round(float(nx.community.modularity(Gk, comms, weight='w')), 3),
               'communities': [dict(id=i, size=len(c), top=[k for k, _ in sorted(((k, kc[k]) for k in c), key=lambda x: -x[1])[:8]],
                                    topic_profile=[round(float(x), 1) for x in ct[i]], dominant_topic=int(np.argmax(ct[i]))) for i, c in enumerate(comms)]}
# figure 8: co-word network (display subgraph: keywords with >= 8 occurrences; labels for the top terms per community)
import matplotlib.patheffects as pe
Gd = Gk.subgraph([n for n in Gk.nodes if kc[n] >= 8]).copy(); Gd.remove_nodes_from(list(nx.isolates(Gd)))
B['coword']['n_drawn'] = Gd.number_of_nodes(); B['coword']['n_drawn_min_occurrences'] = 8
pos = nx.spring_layout(Gd, k=2.6 / np.sqrt(Gd.number_of_nodes()), weight='w', seed=42, iterations=600)
fig, ax = plt.subplots(figsize=(7.08, 6.6)); ax.axis('off')
palette = ['#1B9E77', '#7570B3', '#E7298A', '#66A61E', '#A6761D', '#666666', '#E6AB02', '#1F78B4']  # community palette (distinct from the domain colors)
ew = np.array([d['w'] for _, _, d in Gd.edges(data=True)])
nx.draw_networkx_edges(Gd, pos, ax=ax, width=0.25 + 1.4 * ew / ew.max(), edge_color='#c8c8c8', alpha=0.55)
labels = {}; _ncs = []
for i, c in enumerate(comms):
    nodes = [n for n in c if n in Gd]
    if not nodes: continue
    _ncs.append(nx.draw_networkx_nodes(Gd, pos, nodelist=nodes, node_size=[25 + 7 * kc[n] for n in nodes], node_color=palette[i % len(palette)], alpha=0.88, ax=ax, linewidths=0.6, edgecolors='white'))
    for n in sorted(nodes, key=lambda x: -kc[x])[:8]: labels[n] = n
_texts = [ax.text(pos[n][0], pos[n][1], n, fontsize=8, color='#111', ha='center', va='center', zorder=6) for n in labels]
for t_ in _texts: t_.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
try:
    from adjustText import adjust_text
    _xs = np.array([pos[n][0] for n in Gd.nodes]); _ys = np.array([pos[n][1] for n in Gd.nodes])
    adjust_text(_texts, x=_xs, y=_ys, ax=ax, expand=(1.3, 1.6), force_text=(0.5, 0.8), force_static=(1.0, 1.4), force_explode=(0.4, 0.8), arrowprops=dict(arrowstyle='-', color='#777', lw=0.4), min_arrow_len=4)
except Exception as _e:
    print('adjustText unavailable:', _e)
for i, c in enumerate(comms[:6]):
    if any(n in Gd for n in c): ax.scatter([], [], color=palette[i], s=50, label=f"C{i}: " + ", ".join(B['coword']['communities'][i]['top'][:3]))
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False, title='Louvain communities (three most frequent terms)', title_fontsize=8, columnspacing=1.0, handletextpad=0.4)
ax.set_title(f"Co-word network: {Gd.number_of_nodes()} keywords with \u2265 8 occurrences shown\n(community detection on the {Gk.number_of_nodes()} keywords with \u2265 5 occurrences and \u2265 1 co-occurrence of \u2265 2)", fontsize=8, loc='left', color='#444')
ax.margins(0.06); fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.14)
fig.savefig(f'{FIG}/fig5_coword.png', dpi=300); plt.close(fig)

# figure 9: country collaboration
Gc = G.copy(); keep = [n for n, d in Gc.nodes(data=True) if d['w'] >= 3]; Gc = Gc.subgraph(keep).copy()
fig, ax = plt.subplots(figsize=(7.08, 5.2)); ax.axis('off')
posc = nx.kamada_kawai_layout(Gc, weight=None) if Gc.number_of_nodes() > 2 else nx.circular_layout(Gc)
ewc = np.array([d['w'] for _, _, d in Gc.edges(data=True)]) if Gc.number_of_edges() else np.array([1])
nx.draw_networkx_edges(Gc, posc, ax=ax, width=0.4 + 3 * ewc / ewc.max(), edge_color='#888', alpha=0.6)
sizes = [40 + 6 * Gc.nodes[n]['w'] for n in Gc.nodes]
_pc = nx.draw_networkx_nodes(Gc, posc, node_size=sizes, node_color='#0072B2', alpha=0.85, ax=ax, linewidths=0.8, edgecolors='white')
ax.margins(0.12); fig.canvas.draw()  # fix the data-to-display transform before placing labels
_ct = []
for n, sz in zip(Gc.nodes, sizes):
    r_px = np.sqrt(sz / np.pi) * fig.dpi / 72  # marker radius in pixels
    xd, yd = ax.transData.transform(posc[n]); x_, y_ = ax.transData.inverted().transform((xd, yd + r_px + 2))
    _ct.append(ax.text(x_, y_, f"{n} ({Gc.nodes[n]['w']})", fontsize=8, ha='center', va='bottom', zorder=6))
for t_ in _ct: t_.set_bbox(dict(boxstyle='square,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9))
try:
    from adjustText import adjust_text as _adj
    _adj(_ct, x=np.array([posc[n][0] for n in Gc.nodes]), y=np.array([posc[n][1] for n in Gc.nodes]), ax=ax, expand=(1.1, 1.3), force_text=(0.3, 0.5), force_static=(0.3, 0.5), only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'}, arrowprops=dict(arrowstyle='-', color='#777', lw=0.4), min_arrow_len=6)
except Exception as _e:
    print('adjustText unavailable:', _e)
ax.set_title(f"Country co-authorship: countries with \u2265 3 articles (n = {int(has_c.sum())} articles with country data)\nnode size \u221d articles; edge width \u221d jointly authored articles", fontsize=8, loc='left')
ax.margins(0.12); fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
fig.savefig(f'{FIG}/fig4_countries.png', dpi=300); plt.close(fig)

# productivity table
rows = []
for i in range(15):
    rows.append(dict(rank=i + 1, journal=B['top_journals'][i][0] if i < len(B['top_journals']) else '', journal_n=B['top_journals'][i][1] if i < len(B['top_journals']) else '',
                     country=B['top_countries'][i][0] if i < len(B['top_countries']) else '', country_n=B['top_countries'][i][1] if i < len(B['top_countries']) else '',
                     institution=B['top_institutions'][i][0] if i < len(B['top_institutions']) else '', institution_n=B['top_institutions'][i][1] if i < len(B['top_institutions']) else ''))
pd.DataFrame(rows).to_csv(f'{OUT}/table_productivity.csv', index=False)

# MeSH co-occurrence (supplementary)
MESH_GENERIC = {'Humans', 'Male', 'Female', 'Adult', 'Adolescent', 'Young Adult', 'Child', 'Child, Preschool', 'Middle Aged', 'Aged', 'Animals',  # check tags
                'Baseball', 'Softball', 'Sports', 'Athletes',  # search term and its generic parents
                'United States', 'Japan', 'Taiwan', 'Republic of Korea', 'Universities', 'Students', 'Schools',  # geographic / institutional
                'Cross-Sectional Studies', 'Retrospective Studies', 'Prospective Studies', 'Cohort Studies', 'Case-Control Studies', 'Longitudinal Studies', 'Pilot Projects',
                'Surveys and Questionnaires', 'Analysis of Variance', 'Reproducibility of Results', 'Time Factors', 'Age Factors', 'Sex Factors', 'Follow-Up Studies'}  # study design / statistics
docs_m = [(set(x.strip() for x in str(s).split('|') if x.strip() and x.strip() not in MESH_GENERIC), t) for s, t in zip(en['mesh'], en['topic']) if isinstance(s, str) and s.strip()]
mc = collections.Counter(x for s, _ in docs_m for x in s)
B['mesh_coverage'] = len(docs_m); B['top_mesh'] = mc.most_common(25)
mv = {k for k, v in mc.items() if v >= 5}
Gm = nx.Graph()
for s_, _ in docs_m:
    ss = sorted(x for x in s_ if x in mv)
    for x in ss: Gm.add_node(x, w=mc[x])
    for a, bb in itertools.combinations(ss, 2): Gm.add_edge(a, bb, w=Gm[a][bb]['w'] + 1 if Gm.has_edge(a, bb) else 1)
Gm.remove_edges_from([(a, bb) for a, bb, dd in Gm.edges(data=True) if dd['w'] < 3]); Gm.remove_nodes_from(list(nx.isolates(Gm)))
if Gm.number_of_nodes() > 5:
    cm = sorted(nx.community.louvain_communities(Gm, weight='w', seed=42), key=len, reverse=True)
    posm = nx.spring_layout(Gm, k=2.4 / np.sqrt(Gm.number_of_nodes()), weight='w', seed=42, iterations=600)
    fig, ax = plt.subplots(figsize=(7.08, 7.2)); ax.axis('off')
    ewm = np.array([dd['w'] for _, _, dd in Gm.edges(data=True)])
    nx.draw_networkx_edges(Gm, posm, ax=ax, width=0.3 + 1.2 * ewm / ewm.max(), edge_color='#c8c8c8', alpha=0.55)
    labm = {}; _mcs = []
    for i, c in enumerate(cm):
        _mcs.append(nx.draw_networkx_nodes(Gm, posm, nodelist=list(c), node_size=[30 + 4 * mc[n] for n in c], node_color=palette[i % len(palette)], alpha=0.88, ax=ax, linewidths=0.6, edgecolors='white'))
        for n in sorted(c, key=lambda x: -mc[x])[:8]: labm[n] = n
    _tm = [ax.text(posm[n][0], posm[n][1], n, fontsize=8, color='#111', ha='center', va='center', zorder=6) for n in labm]
    for t_ in _tm: t_.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
    try:
        from adjustText import adjust_text
        adjust_text(_tm, x=np.array([posm[n][0] for n in Gm.nodes]), y=np.array([posm[n][1] for n in Gm.nodes]), ax=ax, expand=(1.3, 1.6), force_text=(0.5, 0.8), force_static=(1.0, 1.4), force_explode=(0.4, 0.8), arrowprops=dict(arrowstyle='-', color='#777', lw=0.4), min_arrow_len=4)
    except Exception as _e:
        print('adjustText unavailable:', _e)
    for i, c in enumerate(cm[:6]):
        ax.scatter([], [], color=palette[i % len(palette)], s=50, label=f"M{i}: " + ", ".join(sorted(c, key=lambda x: -mc[x])[:3]))
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=1, frameon=False, title='Louvain communities (three most frequent headings)', title_fontsize=8, handletextpad=0.4)
    ax.set_title(f"MeSH co-occurrence network: {Gm.number_of_nodes()} headings with \u2265 5 occurrences, edges with \u2265 3 co-occurrences\n(check tags, study-design, geographic, and institutional headings removed)", fontsize=8, loc='left', color='#444')
    ax.margins(0.06); fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.2)
    fig.savefig(f'{FIG}/figS3_mesh.png', dpi=300); plt.close(fig)
    B['mesh_network'] = dict(nodes=Gm.number_of_nodes(), edges=Gm.number_of_edges(), communities=len(cm), modularity=round(float(nx.community.modularity(Gm, cm, weight='w')), 3))
# search date: latest year-month among matched
ym = en[en.matched & en['pub_year'].notna()].copy()
ym['mo'] = ym['pub_month'].astype(str).str[:3].map({m: i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)})
ym['y'] = pd.to_numeric(ym['pub_year'], errors='coerce')
latest = ym.dropna(subset=['y', 'mo']).sort_values(['y', 'mo']).tail(1)
B['latest_pubmed_year_month'] = (int(latest['y'].iloc[0]), int(latest['mo'].iloc[0])) if len(latest) else None
B['latest_year_in_corpus'] = int(df['Publication Year'].max())

# ---------------------------------------------------------------- authors: productivity and co-authorship
arows = []
for _, r in en[en['authors'].notna()].iterrows():
    for a in str(r['authors']).split('|'):
        if a.strip(): arows.append((norm_author(a), r['dom'], int(r['topic'])))
A = pd.DataFrame(arows, columns=['author', 'dom', 'topic'])
acnt = A.groupby('author').size().sort_values(ascending=False)
B['authors'] = dict(distinct=int(len(acnt)), ge2=int((acnt >= 2).sum()), ge5=int((acnt >= 5).sum()),
                    top=[(a, int(n), A[A.author == a].dom.mode().iloc[0]) for a, n in acnt.head(15).items()])
Ga = nx.Graph()
for _, r in en[en['authors'].notna()].iterrows():
    au = sorted({norm_author(a) for a in str(r['authors']).split('|') if a.strip()}); au = [a for a in au if acnt[a] >= 2]
    for a in au: Ga.add_node(a, n=int(acnt[a]), dom=A[A.author == a].dom.mode().iloc[0])
    for a, b2 in itertools.combinations(au, 2): Ga.add_edge(a, b2, w=Ga[a][b2]['w'] + 1 if Ga.has_edge(a, b2) else 1)
comps = sorted(nx.connected_components(Ga), key=len, reverse=True)
within = sum(1 for a, b2 in Ga.edges if Ga.nodes[a]['dom'] == Ga.nodes[b2]['dom'])
B['coauthor_network'] = dict(nodes=Ga.number_of_nodes(), edges=Ga.number_of_edges(), components=len(comps), largest=[len(c) for c in comps[:3]],
                             largest_mix=[dict(collections.Counter(Ga.nodes[n]['dom'] for n in c)) for c in comps[:3]],
                             within_domain_edge_share=round(within / max(Ga.number_of_edges(), 1), 3))
# Supplementary Figure S2: author co-authorship network (authors with >= 2 papers; components with >= 4 authors)
keep = [n for c in comps if len(c) >= 4 for n in c]; Gs = Ga.subgraph(keep).copy()
B['coauthor_network']['prolific_not_drawn'] = [(n, int(Ga.nodes[n]['n']), len(next(c for c in comps if n in c))) for n in Ga.nodes if Ga.nodes[n]['n'] >= 7 and n not in Gs]
pos_a = nx.spring_layout(Gs, k=0.5, weight='w', seed=42, iterations=400)
fig, ax = plt.subplots(figsize=(7.08, 6.2)); ax.axis('off')
nx.draw_networkx_edges(Gs, pos_a, ax=ax, width=0.4, edge_color='#c8c8c8', alpha=0.7)
for dcode in DOMS:
    col = D[dcode]['color']; nodes = [n for n in Gs.nodes if Gs.nodes[n]['dom'] == dcode]
    nx.draw_networkx_nodes(Gs, pos_a, nodelist=nodes, node_size=[14 + 9 * Gs.nodes[n]['n'] for n in nodes], node_color=col, alpha=0.85, ax=ax, linewidths=0.4, edgecolors='white', label=f"dominant domain {dcode}")
lab_nodes = sorted([n for n in Gs.nodes if Gs.nodes[n]['n'] >= 7], key=lambda n: (round(pos_a[n][0], 1), pos_a[n][1]))
import matplotlib.patheffects as pe
_ta = [ax.text(pos_a[n][0], pos_a[n][1], n, fontsize=8, ha='center', va='center', color='#111', zorder=5) for n in lab_nodes]
for t_ in _ta: t_.set_bbox(dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.9))
try:
    from adjustText import adjust_text
    adjust_text(_ta, x=np.array([pos_a[n][0] for n in Gs.nodes]), y=np.array([pos_a[n][1] for n in Gs.nodes]), ax=ax, expand=(1.3, 1.6), force_text=(0.6, 0.9), force_static=(0.5, 0.8), arrowprops=dict(arrowstyle='-', color='#777', lw=0.5), min_arrow_len=4)
except Exception as _e:
    print('adjustText unavailable:', _e)
ax.legend(fontsize=8, frameon=False, loc='lower left')
ax.set_title(f"Author co-authorship network: {Gs.number_of_nodes()} authors with \u2265 2 articles in components of \u2265 4 authors\n(node size \u221d articles; labels: authors with \u2265 7 articles; color: the author's modal domain)", fontsize=8, loc='left', color='#444')
ax.margins(0.06); fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
fig.savefig(f'{FIG}/figS2_coauthors.png', dpi=300); plt.close(fig)
# productivity table: add author columns
pr = pd.read_csv(f'{OUT}/table_productivity.csv')
pr['author'] = [B['authors']['top'][i][0] if i < len(B['authors']['top']) else '' for i in range(len(pr))]
pr['author_n'] = [B['authors']['top'][i][1] if i < len(B['authors']['top']) else '' for i in range(len(pr))]
pr['author_dom'] = [B['authors']['top'][i][2] if i < len(B['authors']['top']) else '' for i in range(len(pr))]
pr.to_csv(f'{OUT}/table_productivity.csv', index=False)
json.dump(B, open(f'{OUT}/biblio.json', 'w'), indent=1, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))
print(json.dumps({k: B[k] for k in ['n', 'matched', 'match_rate', 'match_by_domain', 'country_coverage', 'international_collaboration_rate', 'n_journals', 'citations', 'journal_overlap', 'author_overlap', 'keyword_coverage', 'latest_pubmed_year_month']}, indent=1))
print("top countries:", B['top_countries'][:8]); print("top journals:", B['top_journals'][:8]); print("coword:", {k: v for k, v in B['coword'].items() if k != 'communities'})
for c in B['coword']['communities'][:6]: print("  ", c['id'], c['size'], c['top'][:6], 'dominant topic', c['dominant_topic'])
