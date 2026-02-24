import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client_llm = OpenAI()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client_db = chromadb.PersistentClient(path="./financial_rag_db")
collection = client_db.get_collection(name="card_member_data")


# 1. 파일 읽기 및 공백 제거 
qrels_df = pd.read_csv("qrels_v1.csv")

qrels_df['query_id'] = qrels_df['query_id'].astype(str).str.strip()
qrels_df['doc_id'] = qrels_df['doc_id'].astype(str).str.strip()

qrels_dict = {
    q_id: qrels_df[qrels_df['query_id'] == q_id]['doc_id'].tolist() 
    for q_id in qrels_df['query_id'].unique()
}

# [디버깅 출력] 질문별로 정답이 몇 개씩 로드되었는지 확인
print("\n" + "="*50)
print("[시스템 체크] 정답지(Qrels) 로드 현황")
for q_id in ["q1", "q2_silver", "q2_gourmet", "q2_shopping"]:
    count = len(qrels_dict.get(q_id, []))
    print(f"👉 ID: {q_id:<12} | 정답 수: {count:>3}개 {'✅' if count > 0 else '확인 필요'}")
print("="*50 + "\n")
# qrels_df = pd.read_csv("qrels2.csv")
# qrels_dict = {q_id: qrels_df[qrels_df['query_id'] == q_id]['doc_id'].astype(str).tolist() for q_id in qrels_df['query_id'].unique()}

queries_info = [
    {
        "q_id": "q1", 
        "text": "서울에 거주하는 40대 고객 중 교육비나 학원 관련 지출이 두드러지는 학부모를 찾아줘.",
        "where": {"$and": [{"SIDO": "서울"}, {"AGE": {"$gte": 40}}, {"AGE": {"$lt": 50}}]},
        "llm_prompt": "이 고객의 요약 데이터에 '학원', '서적', '교육'과 관련된 지출이 명시되어 있으면 YES. '유통'이나 '요식업'만 있으면 무조건 NO로 답해."
    },
    
    {
    "q_id": "q2_silver", 
    "text": "경기에 거주하는 5060 세대 중 병원이나 보험 등 의료 관련 지출이 잦은 고객을 찾아줘.",
    "where": {"$and": [{"SIDO": "경기"}, {"AGE": {"$gte": 50}}]}, # 50세 이상 전체
    "llm_prompt": "이 고객의 요약 데이터에 '병원', '의료', '보험', '약국' 관련 지출이 명시되어 있으면 YES. 단순히 '유통' 지출만 있으면 NO로 답해."
    },
    {
    "q_id": "q2_gourmet", 
    "text": "서울에 거주하는 3040 고객 중 식당이나 카페 등 외식 소비가 활발한 미식가 타겟을 추천해줘.",
    "where": {"$and": [{"SIDO": "서울"}, {"AGE": {"$gte": 30}}, {"AGE": {"$lt": 50}}]},
    "llm_prompt": "이 고객의 요약 데이터에 '식당', '카페', '요식', '음식점' 관련 지출이 높다고 명시되어 있으면 YES. 아니면 NO로 답해."
    },
    {
        "q_id": "q2_shopping", 
        "text": "지역에 상관없이 2050 세대 중 백화점이나 대형마트 등 유통업 소비 규모가 큰 쇼핑 우수 고객을 찾아줘.",
        "where": {
            "$and": [
                {"AGE": {"$gte": 20}},
                {"AGE": {"$lt": 60}}
            ]
        },
        "llm_prompt": "이 고객의 요약 데이터에 '유통', '백화점', '마트', '영리' 관련 지출이 지배적이면 YES. 아니면 NO로 답해."
    }
]

# 평가 지표 계산 전용 함수 (Precision, Recall, F1 Score)
def calc_metrics(hits, retrieved_count, true_total):
    precision = (hits / retrieved_count) if retrieved_count > 0 else 0.0
    recall = (hits / true_total) if true_total > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision * 100, recall * 100, f1 * 100

def evaluate_pipeline():
    results_list = []
    retrieve_k = 50  # 50명 검색

    for info in queries_info:
        q_id = info["q_id"]
        true_answers = qrels_dict.get(q_id, [])
        total_true = len(true_answers)  # 정답지에 있는 진짜 정답의 총 개수
        
        query_embedding = model.encode(info["text"]).tolist()

        # Step 0: 무지성 벡터 검색
        res_0 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
        hits_0 = sum(1 for doc in res_0['ids'][0] if doc in true_answers)
        p0, r0, f1_0 = calc_metrics(hits_0, retrieve_k, total_true)

        # Step 1: SQL + 벡터 하이브리드 검색
        res_1 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k, where=info["where"])
        hits_1 = sum(1 for doc in res_1['ids'][0] if doc in true_answers)
        p1, r1, f1_1 = calc_metrics(hits_1, retrieve_k, total_true)

        # Step 2: LLM 의도 필터링
        hits_2 = 0
        passed_ids_2 = []
        for i, doc_id in enumerate(res_1['ids'][0]):
            summary = res_1['documents'][0][i]
            response = client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"조건: {info['llm_prompt']}\n데이터: {summary}\nYES/NO만 답해."}],
                temperature=0.0
            )
            if response.choices[0].message.content.strip().startswith("YES"):
                passed_ids_2.append(doc_id)
                if doc_id in true_answers: hits_2 += 1
                
        p2, r2, f1_2 = calc_metrics(hits_2, len(passed_ids_2), total_true)

        # 결과를 보기 좋게 포맷팅하여 리스트에 추가
        results_list.append({
            "Q": q_id.upper(),
            "Step 0 (단순검색)": f"F1: {f1_0:.1f} (P:{p0:.1f}, R:{r0:.1f})",
            "Step 1 (SQL필터)": f"F1: {f1_1:.1f} (P:{p1:.1f}, R:{r1:.1f})",
            "Step 2 (LLM필터)": f"F1: {f1_2:.1f} (P:{p2:.1f}, R:{r2:.1f})"
        })

    return pd.DataFrame(results_list)

print("\n" + "="*80)
print("[최종 평가 결과] P=정밀도(Precision), R=재현율(Recall), F1=조화평균")
print("="*80)
print(evaluate_pipeline().to_string(index=False))