from common import load_dataset, retrieve_candidates

query = "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘."

df = load_dataset()
candidates = retrieve_candidates(query, df, top_k=10)

print("=== [No Filter 결과] ===")
print(candidates[["summary"]])