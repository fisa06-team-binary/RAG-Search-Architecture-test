import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# 1. .env 파일에서 환경 변수 불러오기
load_dotenv() 

# 키가 제대로 들어왔는지 체크
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("🚨 OPENAI_API_KEY가 세팅되지 않았습니다. .env 파일을 확인하세요!")

client_llm = OpenAI()

# 2. 임베딩 모델 및 DB 로드
print("모델 및 DB 로딩 중...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client_db = chromadb.PersistentClient(path="./financial_rag_db")
collection = client_db.get_collection(name="card_member_data")

# 3. 실험할 질문 세팅 (인천 여행객)
query_text = "인천에 거주하는 젊은 고객 중 숙박·여행 관련 소비가 눈에 띄는 사람을 찾아줘."
where_condition = {"$and": [{"SIDO": "인천"}, {"AGE": {"$lte": 39}}]}

# 4. 1차 검색 (Step 1과 동일)
print("\n🔍 [1차] 벡터 + SQL 검색 수행 중...")
query_embedding = model.encode(query_text).tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where=where_condition
)

# 5. 2차 검색: LLM 기반 의도 필터링
print("\n[2차] LLM 의도 검증 시작 (10명 대상)...\n")
final_passed_data = []

for i in range(len(results['ids'][0])):
    doc_id = results['ids'][0][i]
    summary = results['documents'][0][i]
    
    prompt = f"""
    당신은 금융 데이터 분석가입니다. 사용자는 "숙박 및 여행 관련 소비가 눈에 띄는 고객"을 찾고 있습니다.
    아래 고객의 소비 요약 데이터를 읽고, 이 고객이 여행/숙박(HOTEL, TRVL 등)에 유의미한 지출을 하는 페르소나인지 판단하세요.
    
    [고객 데이터]: {summary}
    
    답변 규칙:
    1. 조건에 부합하면 첫 줄에 'YES', 아니면 'NO'를 작성하세요.
    2. 두 번째 줄에는 그렇게 판단한 이유를 1문장으로 짧게 작성하세요.
    """
    
    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    llm_answer = response.choices[0].message.content.strip()
    decision = "YES" if llm_answer.startswith("YES") else "NO"
    reason = llm_answer.split('\n')[1] if '\n' in llm_answer else llm_answer
    
    print(f"[{doc_id}] LLM 판단: {decision} | 이유: {reason}")
    
    if decision == "YES":
        final_passed_data.append({
            "고객ID": doc_id,
            "데이터 요약": summary,
            "LLM 통과 여부": decision,
            "판단 근거": reason
        })

# 6. 최종 결과 출력
print(f"\n{'='*70}")
print(f"[최종 결과] 1차 필터링 10명 중, LLM 검증을 통과한 {len(final_passed_data)}명")
print(f"{'='*70}")
if final_passed_data:
    df_final = pd.DataFrame(final_passed_data)
    print(df_final.to_string(index=False))
else:
    print("조건에 완벽히 부합하는 고객이 없습니다.")