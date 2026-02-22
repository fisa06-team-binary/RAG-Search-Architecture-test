import os
import pandas as pd
from openai import OpenAI

# =============================
# 0. LLM 설정
# =============================
# 터미널에서 export OPENAI_API_KEY="너의키" 해놔야 함
client = OpenAI()

# =============================
# 1. 데이터 로드 및 전처리
# =============================
df = pd.read_csv("data/우리은행10000_벡터문장.csv")

# AGE 정제
df = df[df["AGE"] != "기타"]
df["AGE"] = df["AGE"].str.replace("대", "", regex=False)
df["AGE"] = df["AGE"].astype(int)

print("전체 데이터 수:", len(df))


# =============================
# 2. Query 정의
# =============================
queries_info = [
    {
        "q_id": "q1",
        "text": "서울에 거주하는 40대 고객 중 교육 관련 지출이 두드러지는 학부모를 찾아줘.",
        "region": "서울",
        "age_min": 40,
        "age_max": 50,
        "llm_prompt": "이 고객의 요약 데이터에 '학원', '서적', '교육' 관련 지출이 명시되어 있으면 YES, 아니면 NO로만 답해."
    },
    {
        "q_id": "q2",
        "text": "경기에 거주하는 50대 고객 중 자동차나 주유 관련 소비가 많은 사람을 찾아줘.",
        "region": "경기",
        "age_min": 50,
        "age_max": 60,
        "llm_prompt": "이 고객의 요약 데이터에 '자동차', '연료', '정비', '주유' 관련 지출이 명시되어 있으면 YES, 아니면 NO로만 답해."
    }
]


# =============================
# 3. LLM 판별 함수
# =============================
def llm_filter(summary, instruction):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 조건 판별기다. 반드시 YES 또는 NO로만 답해."},
            {"role": "user", "content": instruction + "\n\n고객 요약:\n" + summary}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"


# =============================
# 4. 실험 실행
# =============================
for q in queries_info:

    print("\n===================================")
    print("Query:", q["q_id"])
    print("질문:", q["text"])

    # ---------------------------
    # 0차: 검색기 결과 (Top-K 가정)
    # 여기서는 단순히 전체를 Top-K로 가정
    # 실제로는 vector 검색 결과를 넣으면 됨
    # ---------------------------
    df_topk = df.copy()
    print("0차 (검색기 결과 수):", len(df_topk))

    # ---------------------------
    # 1차: SQL / 메타데이터 필터
    # ---------------------------
    df_meta = df_topk[
        (df_topk["HOUS_SIDO_NM"] == q["region"]) &
        (df_topk["AGE"] >= q["age_min"]) &
        (df_topk["AGE"] < q["age_max"])
    ]

    print("1차 (SQL/메타데이터 필터 수):", len(df_meta))

    # ---------------------------
    # 2차: SQL + LLM 필터
    # ---------------------------
    # ⚠ 비용 방지 위해 최대 30개만 LLM 적용
    df_meta_sample = df_meta.head(30)

    df_llm = df_meta_sample[
        df_meta_sample["summary"].apply(
            lambda x: llm_filter(x, q["llm_prompt"])
        )
    ]

    print("2차 (SQL + LLM 필터 수):", len(df_llm))

    print("\n최종 후보 일부:")
    print(df_llm[["DID_SEQ", "AGE", "HOUS_SIDO_NM"]].head())

    