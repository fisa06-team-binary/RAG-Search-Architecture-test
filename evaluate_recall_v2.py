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
qrels_df = pd.read_csv("qrels_v2.csv")

qrels_df['query_id'] = qrels_df['query_id'].astype(str).str.strip()
qrels_df['doc_id'] = qrels_df['doc_id'].astype(str).str.strip()

qrels_dict = {
    q_id: qrels_df[qrels_df['query_id'] == q_id]['doc_id'].tolist() 
    for q_id in qrels_df['query_id'].unique()
}

# [디버깅 출력] 질문별로 정답이 몇 개씩 로드되었는지 확인
print("\n" + "="*50)
print("[시스템 체크] 정답지(Qrels) 로드 현황")
for q_id in ["q1", "q2", "q3", "q4"]:
    count = len(qrels_dict.get(q_id, []))
    print(f"👉 ID: {q_id:<12} | 정답 수: {count:>3}개 {'✅' if count > 0 else '확인 필요'}")
print("="*50 + "\n")

queries_info = [
    {
        "q_id": "q1", 
        "text": "서울에 거주하는 40대 고객 중 교육비나 학원 관련 지출이 두드러지는 학부모를 찾아줘.",
        "where": {"$and": [{"SIDO": "서울"}, {"AGE": {"$gte": 40}}, {"AGE": {"$lt": 50}}]},
        "llm_prompt": "이 고객의 요약 데이터에서 '학원', '서적', '교육'과 관련된 지출 성향이 얼마나 강한지 평가해."
    },
    {
        "q_id": "q2", 
        "text": "경기에 거주하는 5060 세대 중 병원이나 보험 등 의료 관련 지출이 잦은 고객을 찾아줘.",
        "where": {"$and": [{"SIDO": "경기"}, {"AGE": {"$gte": 50}}]},
        "llm_prompt": "이 고객의 요약 데이터에서 '병원', '의료', '보험', '약국'과 관련된 지출 성향이 얼마나 강한지 평가해."
    },
    {
        "q_id": "q3", 
        "text": "서울에 거주하는 3040 고객 중 식당이나 카페 등 외식 소비가 활발한 미식가 타겟을 추천해줘.",
        "where": {"$and": [{"SIDO": "서울"}, {"AGE": {"$gte": 30}}, {"AGE": {"$lt": 50}}]},
        "llm_prompt": "이 고객의 요약 데이터에서 '식당', '카페', '요식', '음식점'과 관련된 외식 지출 성향이 얼마나 강한지 평가해."
    },
    {
        "q_id": "q4", 
        "text": "지역에 상관없이 2050 세대 중 백화점이나 대형마트 등 유통업 소비 규모가 큰 쇼핑 우수 고객을 찾아줘.",
        "where": {"$and": [{"AGE": {"$gte": 20}}, {"AGE": {"$lt": 60}}]},
        "llm_prompt": "이 고객의 요약 데이터에서 '유통', '백화점', '마트', '영리'와 관련된 쇼핑 지출 성향이 얼마나 강한지 평가해."
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

        # Step 0: 단순 벡터 검색
        res_0 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
        hits_0 = sum(1 for doc in res_0['ids'][0] if doc in true_answers)
        p0, r0, f1_0 = calc_metrics(hits_0, retrieve_k, total_true)

        # Step 1: SQL + 벡터 하이브리드 검색
        res_1 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k, where=info["where"])
        hits_1 = sum(1 for doc in res_1['ids'][0] if doc in true_answers)
        p1, r1, f1_1 = calc_metrics(hits_1, retrieve_k, total_true)

        # Step 2: LLM 관련성 점수(0~1) 생성 및 최적 임계값 탐색 (논문 구현)
        doc_scores = []
        for i, doc_id in enumerate(res_1['ids'][0]):
            summary = res_1['documents'][0][i]
            
            # 논문과 같이 0.0에서 1.0 사이의 점수만 반환하도록 프롬프트 수정
            sys_prompt = "너는 고객 데이터의 적합성을 평가하는 시스템이야. 주어진 조건과 고객 데이터를 비교하여 관련성을 0.0에서 1.0 사이의 실수로만 대답해. 부가 설명이나 다른 텍스트는 절대 쓰지 마."
            user_prompt = f"조건: {info['llm_prompt']}\n데이터: {summary}"
            
            try:
                response = client_llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                score_str = response.choices[0].message.content.strip()
                score = float(score_str)
            except Exception as e:
                score = 0.0  # 파싱 에러 발생 시 0점 처리
                
            doc_scores.append((doc_id, score))

        # 임계값(Threshold) 최적화 탐색 (0.1 ~ 0.9)
        best_f1 = -1.0
        best_threshold = 0.0
        best_metrics = (0.0, 0.0, 0.0) # p, r, f1
        
        # 0.1부터 0.9까지 0.1 단위로 테스트
        thresholds = [i * 0.1 for i in range(1, 10)]
        for t in thresholds:
            passed_ids = [d_id for d_id, s in doc_scores if s >= t]
            hits = sum(1 for d_id in passed_ids if d_id in true_answers)
            p_t, r_t, f1_t = calc_metrics(hits, len(passed_ids), total_true)
            
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t
                best_metrics = (p_t, r_t, f1_t)

        # 결과를 보기 좋게 포맷팅하여 리스트에 추가
        p2, r2, f1_2 = best_metrics
        results_list.append({
            "Q": q_id.upper(),
            "Step 0 (단순)": f"F1: {f1_0:.1f} (P:{p0:.1f} R:{r0:.1f})",
            "Step 1 (SQL)": f"F1: {f1_1:.1f} (P:{p1:.1f} R:{r1:.1f})",
            "Step 2 (LLM 최적)": f"F1: {f1_2:.1f} (P:{p2:.1f} R:{r2:.1f})",
            "최적 임계값": f"{best_threshold:.1f}" # 찾은 최적의 임계값 출력
        })

    return pd.DataFrame(results_list)

print("\n" + "="*95)
print("[최종 평가 결과] P=정밀도(Precision), R=재현율(Recall), F1=조화평균")
print("="*95)
print(evaluate_pipeline().to_string(index=False))