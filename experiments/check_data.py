import pandas as pd

df = pd.read_csv("data/우리은행10000_벡터문장.csv")

print("컬럼 목록:")
print(df.columns)

print("\n상위 5개 데이터:")
print(df.head())
