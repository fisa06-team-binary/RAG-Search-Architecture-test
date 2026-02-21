from common import load_dataset, retrieve_candidates

def first_filter(df):
    # '제주' #+ '젊은(20대/30대)' 조건
    mask_region = df["summary"].str.contains("제주")
    mask_young = df["summary"].str.contains("20대|30대")
    return df[mask_region & mask_young]

query = "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘."

df = load_dataset("data/data.csv")
candidates = retrieve_candidates(query, df, top_k=100)
filtered = first_filter(candidates)

print("=== [1차 필터 결과] ===")
print(filtered[["summary"]])
