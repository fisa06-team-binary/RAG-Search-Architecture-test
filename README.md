# 🎯 하이브리드 RAG 기반 금융 타겟 마케팅 최적화 시스템

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-yellow)

## 📌 프로젝트 개요 (Overview)
본 프로젝트는 **금융권 카드 소비 데이터**를 활용하여 초개인화된 타겟 마케팅 시스템을 구축하는 것을 목표로 합니다. 
기존의 단순 텍스트 유사도 기반 RAG(Retrieval-Augmented Generation) 시스템이 가지는 한계(엉뚱한 타겟 추출로 인한 마케팅 비용 낭비 및 고객 피로도 증가)를 극복하기 위해, **SQL 기반의 하드 필터링과 LLM 기반의 의도 검증(Re-ranking)을 결합한 하이브리드 검색 아키텍처**를 설계했습니다. 

그 결과, 4가지 핵심 마케팅 페르소나에서 **최종 추출 고객의 정답률(Precision) 100.0%를 달성**하여 실제 비즈니스에 즉시 투입 가능한 수준의 타겟팅 신뢰도를 검증했습니다.

---

## 🏗 시스템 아키텍처 및 파이프라인 (Architecture)


본 시스템은 효율성과 정확성을 모두 잡기 위해 **3-Step 필터링 파이프라인**으로 작동합니다. (검색 후보군 $k=50$)

1. **Step 0: Vector Search (의미 기반 1차 검색)**
   * `paraphrase-multilingual-MiniLM-L12-v2` 임베딩 모델을 활용해 고객 소비 요약문에서 질문과 의미론적으로 유사한 후보군을 빠르게 추출합니다.
2. **Step 1: Hybrid Search (SQL 메타데이터 필터링)**
   * ChromaDB의 메타데이터 필터(`where` 절)를 활용하여 연령, 거주지 등 '절대 틀리면 안 되는 하드 조건'을 논리 연산자(`$and`, `$gte` 등)로 2차 필터링합니다.
3. **Step 2: LLM Re-ranking (Post-filtering)**
   * 추출된 후보군의 데이터를 `GPT-4o-mini` 모델에 주입하여, "해당 고객의 실제 소비 내역이 마케팅 타겟 조건에 정확히 부합하는지" 최종 검증(YES/NO)하여 오답을 0%로 만듭니다.

---

## 📊 성능 평가 결과 (Performance Evaluation)

4가지 서로 다른 타겟 페르소나를 대상으로 자동화된 성능 평가(`qrels.csv` 기반)를 진행한 결과입니다.

| 마케팅 타겟 페르소나 | Step 0 (단순검색) | Step 1 (SQL필터) | Step 2 (LLM필터) |
| :--- | :---: | :---: | :---: |
| **Q1 (서울 40대 학부모)** | P: 38.0% | P: 44.0% | **P: 100.0%** |
| **Q2_SILVER (경기 5060 의료)** | P: 58.0% | P: 80.0% | **P: 100.0%** |
| **Q2_GOURMET (서울 3040 미식)** | P: 32.0% | P: 52.0% | **P: 100.0%** |
| **Q2_SHOPPING (전지역 2050 쇼핑)** | P: 60.0% | P: 98.0% | **P: 100.0%** |

> 💡 **비즈니스 인사이트 (Precision vs Recall)**
> 제한된 마케팅 예산 내에서 타겟 고객에게 프로모션을 제공하는 금융권 특성상, "모든 잠재 고객을 찾는 것(Recall)"보다 "추출된 고객이 실제 조건에 100% 부합하는가(Precision)"가 훨씬 중요합니다. 본 시스템은 $k=50$이라는 제한된 리소스 환경에서도 오진율 0%를 달성하여 마케팅 ROI를 극대화할 수 있는 토대를 마련했습니다.

---

## 🛠 기술 스택 (Tech Stack)
* **Language:** Python 3.9+
* **Vector Database:** ChromaDB
* **LLM API:** OpenAI API (`gpt-4o-mini`)
* **Embeddings:** `sentence-transformers`
* **Data Processing:** Pandas

---

## 🧗 주요 트러블슈팅 및 회고 (Troubleshooting)

1. **평가 지표의 신뢰도 확보 (정답지 동기화 문제)**
   * **이슈:** 질문 시나리오를 고도화하는 과정에서 검색기가 정답을 찾지 못해 모든 점수가 0.0으로 출력되는 현상 발생.
   * **해결:** 데이터셋 전처리 과정에서 ID의 공백(Whitespace)을 제거(`strip()`)하고 데이터 타입을 문자열로 통일. 정답지 생성기(`make_qrels.py`)와 평가기(`evaluate_recall.py`)의 ID 구조를 동기화하여 자동 평가 시스템의 신뢰도 확보.
2. **ChromaDB 메타데이터 필터링 문법 충돌**
   * **이슈:** `{"AGE": {"$gte": 20, "$lt": 60}}`와 같이 단일 딕셔너리에 다중 연산자를 사용하여 쿼리 파싱 에러 발생.
   * **해결:** `$and` 논리 연산자를 활용하여 쿼리를 명시적으로 분리(`{"$and": [{"AGE": {"$gte": 20}}, {"AGE": {"$lt": 60}}]}`)함으로써 하이브리드 검색의 안정성 확보.
3. **가상환경 경로 및 의존성 관리**
   * **이슈:** 디렉토리 구조 변경 시 파이썬 가상환경(`venv`)의 하드코딩된 경로가 꼬여 라이브러리 인식 불가 현상 발생.
   * **해결:** 의존성 충돌을 방지하기 위해 가상환경을 재구축하고, 배포를 위한 `requirements.txt` 명세화 완료.

---

## 💻 실행 방법 (How to Run)

**1. 환경 설정**
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
