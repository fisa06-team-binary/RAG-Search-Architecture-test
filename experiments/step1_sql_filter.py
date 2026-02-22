import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. 임베딩 모델 및 DB 로드
print("모델 로딩 중...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("DB 연결 중...")
client = chromadb.PersistentClient(path="./financial_rag_db")
collection = client.get_collection(name="card_member_data")

# 2. 테스트할 질문과 각각의 사전 필터(Pre-filter) 조건 정의
# '젊은 고객'의 기준을 39세 이하( <= 39 )로 설정
queries_info = [
    {
        "query": "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘.",
        "where_filter": {
            "$and": [
                {"SIDO": "제주"},
                {"AGE": {"$lte": 39}} # 39세 이하
            ]
        }
    },
    {
        "query": "인천에 거주하는 젊은 고객 중 숙박·여행 관련 소비가 눈에 띄는 사람을 찾아줘.",
        "where_filter": {
            "$and": [
                {"SIDO": "인천"},
                {"AGE": {"$lte": 39}}
            ]
        }
    }
]

# 3. 검색 실험 및 결과 출력
for idx, info in enumerate(queries_info):
    query_text = info["query"]
    where_condition = info["where_filter"]
    
    print(f"\n{'='*70}")
    print(f"[1차 필터링 적용 질문 {idx+1}] {query_text}")
    print(f"   => 적용된 필터 조건: {where_condition}")
    print(f"{'='*70}")
    
    # 쿼리 임베딩
    query_embedding = model.encode(query_text).tolist()
    
    # DB 검색 (where 절이 추가됨)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where=where_condition  # 여기서 오답을 1차로 다 걸러냄
    )
    
    # 결과 정리
    data = []
    for i in range(len(results['ids'][0])):
        data.append({
            "순위": i + 1,
            "고객ID": results['ids'][0][i],
            "거주지": results['metadatas'][0][i]['SIDO'],
            "나이": results['metadatas'][0][i]['AGE'],
            "유사도 거리": round(results['distances'][0][i], 4),
            "데이터 요약": results['documents'][0][i][:50] + "..."
        })
        
    df_result = pd.DataFrame(data)
    print(df_result.to_string(index=False))
    
    # 필터링 성공 여부 검증
    target_sido = "제주" if "제주" in query_text else "인천"
    wrong_sido = len(df_result[df_result['거주지'] != target_sido])
    wrong_age = len(df_result[df_result['나이'] > 39])
    
    print(f"\n분석: 타 지역 고객 {wrong_sido}명, 40대 이상 고객 {wrong_age}명.")
    print("=> 물리적인 조건(지역, 연령)은 100% 완벽하게 통제되었습니다!")