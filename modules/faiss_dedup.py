import faiss
import re, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# Union-Find (Disjoint Set)
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


class Faiss:
    df: pd.DataFrame

    # -----------------------------
    # Params you can tweak
    # -----------------------------
    MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ALPHA_TITLE     = 0.55          # weight for title embeddings
    ALPHA_DESC      = 0.45          # weight for description embeddings
    SIM_THRESHOLD   = 0.95          # cosine threshold to merge
    MAX_K_NEIGHBORS = 64            # FAISS kNN per vector (cap for speed)
    BATCH_SIZE      = 512           # embed batch size

    ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
    model = SentenceTransformer(MODEL_NAME)  # uses GPU if available

    def __init__(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")
        self.df = df
    
    # -----------------------------
    # Normalize text
    # -----------------------------

    def normalize_text(self, s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str).str.strip().str.lower()
        s = s.str.replace("\u0640", "", regex=False)   # tatweel ـ
        s = s.apply(lambda x: self.ARABIC_DIACRITICS.sub("", x))
        s = (s.str.replace("أ","ا").str.replace("إ","ا")
            .str.replace("آ","ا").str.replace("ى","ي")
            .str.replace("ة","ه"))
        s = s.str.replace(r"\s+", " ", regex=True)
        return s

    # -----------------------------
    # Embedding helper
    # -----------------------------


    def embed_weighted(self, titles: list[str], descs: list[str]) -> np.ndarray:
        e_title = self.model.encode(titles, batch_size=self.BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True)
        e_desc  = self.model.encode(descs,  batch_size=self.BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True)
        e = self.ALPHA_TITLE*e_title + self.ALPHA_DESC*e_desc
        # L2-normalize so cosine == dot
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        return e.astype("float32")

    # -----------------------------
    # FAISS clustering per company
    # -----------------------------
    def cluster_company_faiss(self, g: pd.DataFrame, sim_threshold: float = SIM_THRESHOLD, max_k: int = MAX_K_NEIGHBORS):
        n = len(g)
        if n == 1:
            return [[g.index[0]]]

        # embeddings
        emb = self.embed_weighted(g["title_norm"].tolist(), g["description_norm"].tolist())
        d = emb.shape[1]

        # FAISS index with Inner Product (on normalized vectors → cosine)
        index = faiss.IndexFlatIP(d)
        index.add(emb)

        # choose k: include enough neighbors but cap for speed
        k = min(max_k, n)
        sims, nbrs = index.search(emb, k)  # sims: (n,k) similarity matrix

        dsu = DSU(n)
        # Build edges for pairs above threshold
        for i in range(n):
            for j_pos in range(k):
                j = nbrs[i, j_pos]
                if j < 0 or j == i:  # FAISS returns -1 if not filled, and skip self
                    continue
                if sims[i, j_pos] >= sim_threshold:
                    dsu.union(i, j)

        # Collect clusters
        comps = {}
        idx_list = g.index.to_list()
        for i in range(n):
            root = dsu.find(i)
            comps.setdefault(root, []).append(idx_list[i])

        return list(comps.values())


    def dedup(self):
        self.df["company_norm"]     = self.normalize_text(self.df["company"])
        self.df["title_norm"]       = self.normalize_text(self.df["title"])
        self.df["description_norm"] = self.normalize_text(self.df["description"])
        self.df["job_date"]         = pd.to_datetime(self.df["job_date"], errors="coerce")

        records = []
        for comp, g in self.df.groupby("company_norm", sort=False):
            clusters = self.cluster_company_faiss(g)
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
                    "job_ids":      gg["job_id"].tolist()
                })

        dedup_df = (pd.DataFrame(records)
                    .sort_values(["company","last_date"])
                    .reset_index(drop=True))

        # Optional: keep only latest representative rows for downstream joins
        keep_latest = (self.df.assign(_key=self.df["company_norm"])  # placeholder if you want to re-key
                        .loc[:, ["job_id","company","title","description","job_date","source"]])

        return dedup_df, keep_latest