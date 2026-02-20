import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# 0) 로드
# =========================
DATA_PATH = "data/우리은행10000_벡터문장.csv"
df = pd.read_csv(DATA_PATH)

# ✅ 네가 말한 summary 컬럼을 텍스트로 쓸 거라 했으니 여기 고정
TEXT_COL = "summary"

# 필터링에 쓸 컬럼명은 네 CSV에 맞춰 바꿔줘야 함 (예: region, age_group)
REGION_COL = "region"
AGE_COL = "age_group"

# =========================
# 1) BM25 대용: TF-IDF (lexical)
# =========================
tfidf = TfidfVectorizer(min_df=2)
X_tfidf = tfidf.fit_transform(df[TEXT_COL].fillna(""))

def search_tfidf(query: str, candidate_idx: np.ndarray, topk=50):
    qv = tfidf.transform([query])
    sims = cosine_similarity(qv, X_tfidf[candidate_idx]).ravel()
    order = sims.argsort()[::-1][:topk]
    return candidate_idx[order], sims[order]

# =========================
# 2) Vector 검색 (semantic)
# =========================
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(df[TEXT_COL].fillna("").tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)

def search_vector(query: str, candidate_idx: np.ndarray, topk=50):
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = (emb[candidate_idx] @ q).ravel()
    order = sims.argsort()[::-1][:topk]
    return candidate_idx[order], sims[order]

# =========================
# 3) RRF
# =========================
def rrf_fuse(ranklist_a, ranklist_b, k=60, topk=50):
    score = {}
    for r, doc in enumerate(ranklist_a, start=1):
        score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    for r, doc in enumerate(ranklist_b, start=1):
        score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    fused = sorted(score.items(), key=lambda x: x[1], reverse=True)[:topk]
    return np.array([d for d, _ in fused]), np.array([s for _, s in fused])

# =========================
# 4) 평가 (Precision@K)
# =========================
def precision_at_k(retrieved_idx: np.ndarray, qrels_set: set, k=10):
    top = retrieved_idx[:k]
    rel = sum(1 for i in top if i in qrels_set)
    return rel / k

# =========================
# 5) 실험 실행 함수
# =========================
def run_exp1(query: str, qrels_set: set):
    all_idx = np.arange(len(df))

    # no filtering
    idx_nf, _ = search_vector(query, all_idx, topk=50)  # 실험1은 보통 vector로 “필터링 효과” 보여주기 좋음
    p_nf = precision_at_k(idx_nf, qrels_set, k=10)

    # 1차 필터 (예: region)
    idx_f1 = df.index[df[REGION_COL].astype(str).str.contains("제주", na=False)].to_numpy()
    idx_f1_rank, _ = search_vector(query, idx_f1, topk=50)
    p_f1 = precision_at_k(idx_f1_rank, qrels_set, k=10)

    # 2차 필터 (예: region + age)
    idx_f2 = df.index[
        (df[REGION_COL].astype(str).str.contains("제주", na=False)) &
        (df[AGE_COL].isin(["20s", "30s", "20대", "30대"]))
    ].to_numpy()
    idx_f2_rank, _ = search_vector(query, idx_f2, topk=50)
    p_f2 = precision_at_k(idx_f2_rank, qrels_set, k=10)

    return p_nf, p_f1, p_f2

def run_exp2(query: str, qrels_set: set):
    all_idx = np.arange(len(df))

    # noRRF: 예시로 vector-only
    idx_vec, _ = search_vector(query, all_idx, topk=50)
    p_vec = precision_at_k(idx_vec, qrels_set, k=10)

    # yesRRF: TF-IDF + Vector를 RRF로 결합
    idx_t, _ = search_tfidf(query, all_idx, topk=50)
    fused_idx, _ = rrf_fuse(idx_t, idx_vec, k=60, topk=50)
    p_rrf = precision_at_k(fused_idx, qrels_set, k=10)

    return p_vec, p_rrf

# =========================
# 6) qrels 로드(예시)
# =========================
# qrels 파일은 “정답 고객 row index” 또는 “고객ID”를 담고 있어야 함.
# 가장 쉬운 건: CSV에 row index 기준으로 qrels 만들기.
def load_qrels(path: str) -> set:
    q = pd.read_csv(path)
    # qrels 컬럼명 예: relevant_idx
    return set(q["relevant_idx"].astype(int).tolist())

# =========================
# 7) 실행 예시
# =========================
if __name__ == "__main__":
    # 너가 만든 질문들
    q1 = "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘."
    q2 = "인천에 거주하는 젊은 고객 중 숙박·여행 관련 소비가 눈에 띄는 사람을 찾아줘."

    # qrels 파일 경로(네가 만들어야 함)
    qrels1 = load_qrels("qrels/exp1_q1_qrels.csv")
    qrels2 = load_qrels("qrels/exp1_q2_qrels.csv")

    p_nf, p_f1, p_f2 = run_exp1(q1, qrels1)
    print("[실험1-Q1] P@10 no-filter:", p_nf, "filter1:", p_f1, "filter1+2:", p_f2)

    p_nf, p_f1, p_f2 = run_exp1(q2, qrels2)
    print("[실험1-Q2] P@10 no-filter:", p_nf, "filter1:", p_f1, "filter1+2:", p_f2)

    # 실험2 예시 qrels도 동일 방식으로 준비
    q3 = "서울에 거주하는 30대 중 외식 소비가 많은 고객을 추천해줘."
    qrels3 = load_qrels("qrels/exp2_q1_qrels.csv")

    p_no, p_yes = run_exp2(q3, qrels3)
    print("[실험2-Q1] P@10 noRRF:", p_no, "yesRRF:", p_yes)