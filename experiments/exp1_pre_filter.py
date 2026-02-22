import pandas as pd

# 1. 데이터 로드
df = pd.read_csv("data/우리은행10000_벡터문장.csv")

# 2. AGE 정제
df = df[df["AGE"] != "기타"]   # 기타 제거
df["AGE"] = df["AGE"].str.replace("대", "", regex=False)
df["AGE"] = df["AGE"].astype(int)

print("전체 데이터 개수:", len(df))

# -------------------------------------------------
# 실험 1 - 질문 1
# -------------------------------------------------

df_jeju = df[df["HOUS_SIDO_NM"] == "제주"]
df_jeju_young = df_jeju[(df_jeju["AGE"] >= 20) & (df_jeju["AGE"] < 40)]

print("제주 + 20~39세 고객 수:", len(df_jeju_young))

threshold = df_jeju_young["TOT_USE_AM"].quantile(0.8)
df_trend = df_jeju_young[df_jeju_young["TOT_USE_AM"] >= threshold]

print("상위 20% 소비 고객 수:", len(df_trend))
print(df_trend[["DID_SEQ", "AGE", "HOUS_SIDO_NM", "TOT_USE_AM"]].head())

# -------------------------------------------------
# 실험 1 - 질문 2
# -------------------------------------------------

df_incheon = df[df["HOUS_SIDO_NM"] == "인천"]
df_incheon_young = df_incheon[(df_incheon["AGE"] >= 20) & (df_incheon["AGE"] < 40)]

print("\n인천 + 20~39세 고객 수:", len(df_incheon_young))

# 반드시 copy()
df_incheon_young = df_incheon_young.copy()

# 여행 소비 계산
df_incheon_young["TRAVEL_TOTAL"] = (
    df_incheon_young["HOTEL_AM"] + df_incheon_young["TRVL_AM"]
)

# 🔥 0원 제거
df_travel_positive = df_incheon_young[df_incheon_young["TRAVEL_TOTAL"] > 0]

print("여행 소비 0 초과 고객 수:", len(df_travel_positive))

# 상위 20%
threshold2 = df_travel_positive["TRAVEL_TOTAL"].quantile(0.8)

df_travel_top = df_travel_positive[
    df_travel_positive["TRAVEL_TOTAL"] >= threshold2
]

print("여행 소비 상위 20% 고객 수:", len(df_travel_top))
print(df_travel_top[["DID_SEQ", "AGE", "HOUS_SIDO_NM", "TRAVEL_TOTAL"]].head())

print("\n===== 실험 1 요약 =====")
print("전체 데이터:", len(df))
print("제주 후보:", len(df_jeju))
print("제주+젊은:", len(df_jeju_young))
print("제주 상위20%:", len(df_trend))

print("\n인천 후보:", len(df_incheon))
print("인천+젊은:", len(df_incheon_young))
print("여행>0:", len(df_travel_positive))
print("여행 상위20%:", len(df_travel_top))

print("\n===== Reduction Rate =====")
print("제주 최종 비율:", len(df_trend) / len(df) * 100, "%")
print("인천 최종 비율:", len(df_travel_top) / len(df) * 100, "%")