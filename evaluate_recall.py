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

qrels_df = pd.read_csv("qrels.csv")
qrels_dict = {q_id: qrels_df[qrels_df['query_id'] == q_id]['doc_id'].astype(str).tolist() for q_id in qrels_df['query_id'].unique()}

queries_info = [
    {
        "q_id": "q1", 
        "text": "서울에 거주하는 40대 고객 중 교육비나 학원 관련 지출이 두드러지는 학부모를 찾아줘.",
        "where": {"$and": [{"SIDO": "서울"}, {"AGE": {"$gte": 40}}, {"AGE": {"$lt": 50}}]},
        "llm_prompt": "이 고객의 요약 데이터에 '학원', '서적', '교육'과 관련된 지출이 명시되어 있으면 YES. '유통'이나 '요식업'만 있으면 무조건 NO로 답해."
    },
    {
        "q_id": "q2", 
        "text": "경기에 거주하는 50대 고객 중 자동차나 주유 관련 소비가 많은 사람을 찾아줘.",
        "where": {"$and": [{"SIDO": "경기"}, {"AGE": {"$gte": 50}}, {"AGE": {"$lt": 60}}]},
        "llm_prompt": "이 고객의 요약 데이터에 '자동차', '연료', '정비', '주유' 관련 지출이 명시되어 있으면 YES. 그 외에는 NO로 답해."
    }
]

# 💡 평가 지표 계산 전용 함수 (Precision, Recall, F1 Score)
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
    retrieve_k = 50  # 20명 검색

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
print("🚀 [최종 평가 결과] P=정밀도(Precision), R=재현율(Recall), F1=조화평균")
print("="*80)
print(evaluate_pipeline().to_string(index=False))