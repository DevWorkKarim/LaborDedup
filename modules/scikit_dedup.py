import os, re, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('data/2025-08-16_2025-08-30.csv')
original_count = len(df)

df = df[(~df['company'].isna()) & (~df['title'].isna())]

# -----------------------------
# 0) Params you can tweak
# -----------------------------
MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ALPHA_TITLE     = 0.55          # weight for title embeddings
ALPHA_DESC      = 0.45          # weight for description embeddings
SIM_THRESHOLD   = 0.95          # cosine threshold to merge
MAX_K_NEIGHBORS = 64            # kNN per vector (cap for speed)
BATCH_SIZE      = 512           # embed batch size

# (stable defaults on macOS / Apple Silicon)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Load model (CPU is safest; switch to "mps" later if you want)
model = SentenceTransformer(MODEL_NAME, device="cpu")

# -----------------------------
# 1) Normalize text
# -----------------------------
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip().str.lower()
    s = s.str.replace("\u0640", "", regex=False)   # tatweel ـ
    s = s.apply(lambda x: ARABIC_DIACRITICS.sub("", x))
    s = (s.str.replace("أ","ا").str.replace("إ","ا")
           .str.replace("آ","ا").str.replace("ى","ي")
           .str.replace("ة","ه"))
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

df["company_norm"]     = normalize_text(df["company"])
df["title_norm"]       = normalize_text(df["title"])
df["description_norm"] = normalize_text(df["description"])
df["job_date"]         = pd.to_datetime(df["job_date"], errors="coerce")

# -----------------------------
# 2) Embedding helper
# -----------------------------
def embed_weighted(titles: list[str], descs: list[str]) -> np.ndarray:
    e_title = model.encode(
        titles, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True
    )
    e_desc  = model.encode(
        descs,  batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True
    )
    e = ALPHA_TITLE*e_title + ALPHA_DESC*e_desc
    # L2-normalize so cosine == dot
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    # sklearn expects float64 or float32; keep float32 and contiguous
    return np.ascontiguousarray(e.astype("float32"))

# -----------------------------
# 3) Union-Find (Disjoint Set)
# -----------------------------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# -----------------------------
# 4) sklearn kNN clustering per company (cosine)
# -----------------------------
def cluster_company_nn(
    g: pd.DataFrame,
    sim_threshold: float = SIM_THRESHOLD,
    max_k: int = MAX_K_NEIGHBORS
):
    n = len(g)
    if n == 1:
        return [[g.index[0]]]

    emb = embed_weighted(g["title_norm"].tolist(), g["description_norm"].tolist())
    k = min(max_k, n)

    # brute-force + cosine distance (stable in high dimensions)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, nbrs = nn.kneighbors(emb, return_distance=True)  # shapes (n,k), (n,k)

    dsu = DSU(n)
    sims = 1.0 - dists  # convert cosine distance → similarity in [0,1]
    for i in range(n):
        for j_pos in range(k):
            j = int(nbrs[i, j_pos])
            if j == i:
                continue
            if float(sims[i, j_pos]) >= sim_threshold:
                dsu.union(i, j)

    comps, idx_list = {}, g.index.to_list()
    for i in range(n):
        r = dsu.find(i)
        comps.setdefault(r, []).append(idx_list[i])
    return list(comps.values())

# -----------------------------
# 5) Run over companies & build dedup table
# -----------------------------
records = []
for comp, g in df.groupby("company_norm", sort=False):
    clusters = cluster_company_nn(g)
    for cl in clusters:
        gg = g.loc[cl].sort_values("job_date")
        records.append({
            "company":      gg.iloc[-1]["company"],
            "title":        gg.iloc[-1]["title"],
            "description":  gg.iloc[-1]["description"],
            "first_date":   gg["job_date"].min(),
            "last_date":    gg["job_date"].max(),
            "n_posts":      len(gg),
            "n_sources":    gg["source"].nunique(),
            "sources_list": sorted(gg["source"].unique()),
            "job_ids":      gg["job_id"].tolist(),
        })

dedup_df = (
    pd.DataFrame(records)
      .sort_values(["company", "last_date"])
      .reset_index(drop=True)
)

# Optional: keep only latest representative rows for downstream joins
keep_latest = (
    df.assign(_key=df["company_norm"])
      .loc[:, ["job_id","company","title","description","job_date","source"]]
)


print(dedup_df.head(20).to_string())