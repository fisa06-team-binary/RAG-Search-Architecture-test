print(df["AGE"].head())
print(df["AGE"].dtype)

import pandas as pd

# 1. 데이터 로드
df = pd.read_csv("data/우리은행10000_벡터문장.csv")

print("전체 데이터 개수:", len(df))

# -------------------------------------------------
# 실험 1 - 질문 1
# "제주에서 소비 트렌드를 주도하는 젊은 고객"
# -------------------------------------------------

# 1차 구조 필터: 지역 = 제주
df_jeju = df[df["HOUS_SIDO_NM"] == "제주"]

# 1차 구조 필터: 연령 20~39세
df_jeju_young = df_jeju[(df_jeju["AGE"] >= 20) & (df_jeju["AGE"] < 40)]

print("제주 + 20~39세 고객 수:", len(df_jeju_young))

# 트렌드 주도 = 총 소비 상위 20%
threshold = df_jeju_young["TOT_USE_AM"].quantile(0.8)

df_trend = df_jeju_young[df_jeju_young["TOT_USE_AM"] >= threshold]

print("상위 20% 소비 고객 수:", len(df_trend))
print(df_trend[["DID_SEQ", "AGE", "HOUS_SIDO_NM", "TOT_USE_AM"]].head())

