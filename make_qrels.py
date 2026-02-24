# import pandas as pd

# file_path = 'data/dataset.csv' # 파일명 확인
# df = pd.read_csv(file_path)
# df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').fillna(0)
# qrels_data = []

# # [질문 1] 제주 + 트렌드
# jeju_young = df[(df['HOUS_SIDO_NM'] == '서울') & (df['AGE'] <= 39) & (df['AGE'] > 0)].copy()
# # 핵심: 문장 안에 트렌디한 지출(요식, 문화, 레저 등) 관련 단어가 있는 사람만 정답으로!
# jeju_answers = jeju_young[jeju_young['summary'].str.contains('요식|휴게|문화|레저|쇼핑', na=False)].head(50)

# for _, row in jeju_answers.iterrows():
#     qrels_data.append({"query_id": "q1", "doc_id": str(row['DID_SEQ'])})

# # [질문 2] 인천 + 숙박/여행
# incheon_young = df[(df['HOUS_SIDO_NM'] == '서울') & (df['AGE'] <= 39) & (df['AGE'] > 0)].copy()
# # 핵심: 문장 안에 여행/숙박 관련 단어가 명시된 사람만 정답으로! (LLM이 알아볼 수 있게)
# incheon_answers = incheon_young[incheon_young['summary'].str.contains('숙박|여행|호텔|HOTEL|TRVL|여객', na=False, case=False)].head(50)

# for _, row in incheon_answers.iterrows():
#     qrels_data.append({"query_id": "q2", "doc_id": str(row['DID_SEQ'])})

# pd.DataFrame(qrels_data).to_csv("qrels.csv", index=False, encoding='utf-8-sig')
# print("텍스트 기반으로 튜닝된 완벽한 Qrels 생성 완료!")

import pandas as pd

file_path = 'data/dataset.csv' 
df = pd.read_csv(file_path)
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').fillna(0)
qrels_data = []

# [Q1 정답] 서울 40대 + 학원/교육 지출
seoul_40 = df[(df['HOUS_SIDO_NM'] == '서울') & (df['AGE'] >= 40) & (df['AGE'] < 50)]
q1_answers = seoul_40[seoul_40['summary'].str.contains('학원|교육|서적|OFFEDU|ACDM', na=False, case=False)]
for _, row in q1_answers.iterrows():
    qrels_data.append({"query_id": "q1", "doc_id": str(row['DID_SEQ'])})

# [Q2 정답] 경기 50대 + 자동차/주유 지출
gg_50 = df[(df['HOUS_SIDO_NM'] == '경기') & (df['AGE'] >= 50) & (df['AGE'] < 60)]
q2_answers = gg_50[gg_50['summary'].str.contains('자동차|주유|연료|AUTO|FUEL', na=False, case=False)]
for _, row in q2_answers.iterrows():
    qrels_data.append({"query_id": "q2", "doc_id": str(row['DID_SEQ'])})

# [Q2_SILVER] 경기 5060 + 의료/보험
q2_silver_cond = (df['HOUS_SIDO_NM'] == '경기') & (df['AGE'] >= 50)
q2_silver_answers = df[q2_silver_cond & df['summary'].str.contains('병원|의료|보험|약국|HOSP|INSU', na=False, case=False)]
for _, row in q2_silver_answers.iterrows():
    qrels_data.append({"query_id": "q2_silver", "doc_id": str(row['DID_SEQ'])})

# [Q2_GOURMET] 서울 3040 + 외식/미식
q2_gourmet_cond = (df['HOUS_SIDO_NM'] == '서울') & (df['AGE'] >= 30) & (df['AGE'] < 50)
q2_gourmet_answers = df[q2_gourmet_cond & df['summary'].str.contains('식당|카페|요식|음식점|FSBZ|CAFE', na=False, case=False)]
for _, row in q2_gourmet_answers.iterrows():
    qrels_data.append({"query_id": "q2_gourmet", "doc_id": str(row['DID_SEQ'])})

# [Q2_SHOPPING] 전지역 2050 + 유통/쇼핑
q2_shopping_cond = (df['AGE'] >= 20) & (df['AGE'] < 60)
q2_shopping_answers = df[q2_shopping_cond & df['summary'].str.contains('유통|백화점|마트|영리|DIST|MART', na=False, case=False)]
for _, row in q2_shopping_answers.iterrows():
    qrels_data.append({"query_id": "q2_shopping", "doc_id": str(row['DID_SEQ'])})

# CSV 저장 (BOM 추가로 한글 깨짐 방지)
pd.DataFrame(qrels_data).to_csv("qrels2.csv", index=False, encoding='utf-8-sig')

print(f"총 {len(qrels_data)}개의 정답을 포함한 'qrels2.csv' 생성 완료!")
