import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# 1. 데이터 로드
file_path = 'data/dataset.csv' 
df = pd.read_csv(file_path)

# 2. 임베딩 모델 로드
print("모델 로딩 중...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 3. ChromaDB 설정
client = chromadb.PersistentClient(path="./financial_rag_db")
collection = client.get_or_create_collection(name="card_member_data")

# 4. 데이터 및 메타데이터 준비
documents = df['summary'].astype(str).tolist()
ids = df['DID_SEQ'].astype(str).tolist()

print("메타데이터 처리 중...")
metadatas = []
for i, row in df.iterrows():
    try:
        age_val = int(float(row['AGE']))
    except (ValueError, TypeError):
        age_val = 0
    metadatas.append({
        "SIDO": str(row['HOUS_SIDO_NM']),
        "AGE": age_val,
        "SEX": "여성" if str(row['SEX_CD']) == '2' else "남성",
        "LIFE_STAGE": str(row['LIFE_STAGE'])
    })

# 5. 임베딩 생성
print("데이터 임베딩 생성 중 (약 2~5분 소요)...")
embeddings = model.encode(documents, show_progress_bar=True).tolist()

# 6. 데이터를 쪼개서 DB에 추가 (Batching)
print("DB 저장 중 (배치 처리)...")
batch_size = 2000  # 안전하게 2000개씩 끊어서 저장
for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    collection.add(
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx]
    )
    print(f"진행 상황: {end_idx} / {len(ids)} 건 저장 완료")

print("-" * 30)
print(f"성공: 총 {collection.count()}건의 데이터가 DB에 저장되었습니다.")
print("-" * 30)