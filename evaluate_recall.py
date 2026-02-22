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

# queries_info = [
#     {
#         "q_id": "q1", 
#         "text": "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘.",
#         "where": {"$and": [{"SIDO": "제주"}, {"AGE": {"$lte": 39}}]},
#         "llm_prompt": "이 고객의 데이터에 '문화', '레저', '쇼핑' 같은 여가/트렌드 지출이 명시되어 있으면 YES. 하지만 단순히 '요식업(식비)'이나 '편의점' 지출만 높다면 평범한 생활비 지출이므로 무조건 NO라고 답해."
#     },
#     {
#         "q_id": "q2", "text": "인천에 거주하는 젊은 고객 중 숙박·여행 관련 소비가 눈에 띄는 사람을 찾아줘.",
#         "where": {"$and": [{"SIDO": "인천"}, {"AGE": {"$lte": 39}}]},
#         "llm_prompt": "이 고객의 요약 데이터에 숙박(호텔), 여행 관련 지출이 명시되어 있으면 YES, 아니면 NO"
#     }
# ]

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

def evaluate_pipeline():
    results_list = []
    retrieve_k = 20

    for info in queries_info:
        q_id = info["q_id"]
        true_answers = qrels_dict.get(q_id, [])
        query_embedding = model.encode(info["text"]).tolist()

        # Step 0: 무지성 벡터 검색
        res_0 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
        hits_0 = sum(1 for doc in res_0['ids'][0] if doc in true_answers)
        acc_0 = (hits_0 / retrieve_k) * 100

        # Step 1: SQL + 벡터 검색
        res_1 = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k, where=info["where"])
        hits_1 = sum(1 for doc in res_1['ids'][0] if doc in true_answers)
        acc_1 = (hits_1 / retrieve_k) * 100

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
                
        # LLM이 걸러낸 후 살아남은 사람 중 진짜 정답의 비율
        acc_2 = (hits_2 / len(passed_ids_2)) * 100 if passed_ids_2 else 0.0

        results_list.append({
            "질문": q_id,
            "Step 0 (필터 없음)": f"{acc_0:.1f}%",
            "Step 1 (SQL 적용)": f"{acc_1:.1f}%",
            "Step 2 (LLM 적용)": f"{acc_2:.1f}%"
        })

    return pd.DataFrame(results_list)

print(evaluate_pipeline().to_string(index=False))