import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. 임베딩 모델 로드
print("모델 로딩 중...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. ChromaDB 로드
print("DB 연결 중...")
client = chromadb.PersistentClient(path="./financial_rag_db")
collection = client.get_collection(name="card_member_data")

# 3. 테스트할 질문 리스트
queries = [
    "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘.",
    "인천에 거주하는 젊은 고객 중 숙박·여행 관련 소비가 눈에 띄는 사람을 찾아줘."
]

# 4. 검색 실험 및 결과 출력
for idx, query in enumerate(queries):
    print(f"\n{'='*60}")
    print(f"[질문 {idx+1}] {query}")
    print(f"{'='*60}")
    
    # 쿼리를 임베딩(벡터화)
    query_embedding = model.encode(query).tolist()
    
    # DB에서 필터 없이 단순 유사도 검색 (Top 10)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    
    # 결과를 보기 좋게 Pandas DataFrame으로 정리
    data = []
    for i in range(len(results['ids'][0])):
        data.append({
            "순위": i + 1,
            "고객ID": results['ids'][0][i],
            "거주지(SIDO)": results['metadatas'][0][i]['SIDO'],
            "나이(AGE)": results['metadatas'][0][i]['AGE'],
            "유사도 거리": round(results['distances'][0][i], 4),
            "데이터 요약 요약": results['documents'][0][i][:50] + "..." # 너무 길면 잘라서 표기
        })
        
    df_result = pd.DataFrame(data)
    print(df_result.to_string(index=False))
    
    # 오답률 자동 계산 
    if "제주" in query:
        wrong_count = len(df_result[df_result['거주지(SIDO)'] != '제주'])
        print(f"\n분석: 제주 거주자가 아닌 오답이 10명 중 {wrong_count}명 섞여 있습니다!")
    elif "인천" in query:
        wrong_count = len(df_result[df_result['거주지(SIDO)'] != '인천'])
        print(f"\n분석: 인천 거주자가 아닌 오답이 10명 중 {wrong_count}명 섞여 있습니다!")