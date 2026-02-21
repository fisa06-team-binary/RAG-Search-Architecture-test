# src/exp1_filter_1st_2nd.py
from common import load_dataset, retrieve_candidates

def first_filter(df):
    mask_region = df["summary"].str.contains("제주")
    #mask_young = df["summary"].str.contains("20대|30대")
    return df[mask_region] #& mask_young]

def second_filter_soft(df, query):
    # relevance 대용: 질문 키워드 포함 여부
    keywords = ["소비", "트렌드", "주도"]
    mask = df["summary"].apply(lambda x: any(k in x for k in keywords))
    return df[mask]

query = "제주에서 소비 트렌드를 주도하는 젊은 고객을 찾아줘."

df = load_dataset("data/data.csv")
candidates = retrieve_candidates(query, df, top_k=100)

filtered_1st = first_filter(candidates)
filtered_2nd = second_filter_soft(filtered_1st, query)

print("=== [1차 필터] ===")
print(filtered_1st[["summary"]].head(5))

print("\n=== [1차 + 2차 필터] ===")
print(filtered_2nd[["summary"]].head(5))


###############
# pip install openai
# from openai import OpenAI

# client = OpenAI(api_key="YOUR_API_KEY")

# def llm_relevance_score(query, passage):
#     prompt = f"""
#     Query: {query}
#     Passage: {passage}

#     Score the relevance between 0 and 1 (only return a number):
#     """
#     resp = client.responses.create(
#         model="gpt-4.1-mini",
#         input=prompt
#     )
#     score_text = resp.output_text.strip()
#     return float(score_text)