# Bank Customer Churn — From XAI Diagnosis to Retention Execution

> **2026 Sinchon Union Data·AI Hackathon — Team 16**
> Predicting churn, explaining *why* with SHAP, and validating the *feasibility* of intervention through counterfactual and ROI analysis.

![Model](https://img.shields.io/badge/model-CatBoost%20%C2%B7%20LightGBM%20%C2%B7%20XGBoost-1D4ED8)
![CV PR-AUC](https://img.shields.io/badge/CV%20PR--AUC-0.7312-2563EB)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8895-2563EB)
![Explainability](https://img.shields.io/badge/explainability-SHAP-1E3A8A)

**Live deliverables**

| | Link |
|---|---|
| 📊 Interactive Retention Dashboard | **https://subin21cc.github.io/2026-sinchon-hackathon-team16/** |
| 📄 XAI Deep-Dive Analysis Report | **https://subin21cc.github.io/2026-sinchon-hackathon-team16/report.html** |

---

## Overview

This project tackles bank customer churn not as a pure prediction task, but as an **end-to-end decision pipeline**: predict high-risk customers with a gradient-boosting ensemble, explain each prediction with SHAP, translate those explanations into distinct customer archetypes, and prescribe tailored retention actions — down to LLM-generated outreach emails. Two analyses (counterfactual and ROI targeting) quantify whether the recommended interventions are actually *effective* and *economical*, closing the loop from insight to action.

> **한글 정리** — 이탈 예측을 단순 분류가 아니라 "예측 → 설명(SHAP) → 세그먼트 진단 → 맞춤 리텐션 실행"의 의사결정 파이프라인으로 접근했습니다. 반사실·ROI 분석으로 개입의 **효과와 경제성**까지 검증해 인사이트가 실제 행동으로 이어지도록 설계했습니다.

---

## Key Results

| Metric | Value | Note |
|---|---|---|
| CV PR-AUC | **0.7312** | Official metric (class imbalance 3.7 : 1) |
| ROC-AUC | 0.8895 | Reference |
| Churn rate | 21.2% | 33,007 customers evaluated |
| High-risk pool | 5,076 | Predicted probability > 0.5 |
| Top-20% targeting | **59%** of churners captured | At 1/5 of the marketing budget |
| Counterfactual (reactivation) | **−26 %p** churn probability | Segment-average, inactive high-risk group |

> **한글 정리** — 공식지표 PR-AUC 0.7312(ROC-AUC 0.8895). 위험 상위 20%만 접촉해도 이탈자의 59%를 포착하고, 비활성 고위험군을 재활성화하면 이탈확률이 평균 26%p 낮아집니다.

---

## Methodology

### 1. Data & Feature Engineering
Synthetic bank-churn dataset (Geography, Gender, Age, Balance, NumOfProducts, IsActiveMember, EstimatedSalary, …). We engineered ~22 features, including a **product-count bin**, and age/activity interaction terms, while excluding the `id` identifier from the model per competition rules.

### 2. Model
A **CatBoost · LightGBM · XGBoost** ensemble with 3-seed bagging and optimized weighted blending, validated by 5-fold cross-validation and submitting calibrated probabilities.

| Stage | CV PR-AUC |
|---|---|
| Single CatBoost | 0.7302 |
| + Feature engineering (22 features) | 0.7308 |
| 3 models × 3-seed bagging | 0.7310 |
| Weighted blend optimization | 0.7312 |
| Stacking comparison → adopted | **0.7312** |

### 3. Explainability (SHAP)
Global mean-|SHAP| ranks the **product-count bin** first, exactly matching the EDA finding of a "3+ products cliff." Local SHAP drives the per-customer diagnosis in the dashboard.

### 4. Counterfactual & ROI
- **Counterfactual** — flipping `IsActiveMember` 0 → 1 lowers predicted churn by **−23.1 %p** (individual) and **−26.0 %p** (segment average), grounding the reactivation strategy.
- **ROI targeting** — ranking by risk and contacting the top-K% (assumptions: ₩3,000 per contact, ₩300,000 saved-customer value, 30% campaign success):

  | Contact range | Churners captured | Expected net profit |
  |---|---|---|
  | Top 10% | 37.4% | +₩13.61M |
  | **Top 20% (recommended)** | **59.2%** | **+₩54.65M** |
  | Top 30% | 74.9% | +₩111.56M |

> **한글 정리** — 파생변수 22개(핵심: 상품수 구간)와 CatBoost·LightGBM·XGBoost 앙상블(3시드 배깅·가중 블렌딩, 5-fold CV)로 0.7312를 달성했습니다. 전역 SHAP 1순위는 '상품수 구간'으로 EDA의 '상품 3개↑ 절벽'과 일치합니다. 반사실 분석으로 재활성화 효과(−26%p)를, ROI 시뮬레이션으로 상위 20% 접촉의 타겟팅 효율(이탈자 59% 포착)을 정량화했습니다.

---

## Business Solution — Diagnosis → Prescription

SHAP reveals that high-risk customers fail for **different reasons**, so they receive **different prescriptions**:

| Archetype | Dominant driver | Tailored retention strategy |
|---|---|---|
| Product-overload | Number of products (3+) | Consolidated premium account · fee waiver · product review |
| Middle-aged | Age (50s peak) | Life-stage products (retirement/wealth) · dedicated advisory |
| Germany | Geography · high balance | Local competitive rates · country-specific campaign |
| Inactive | Inactivity | App re-activation · auto-transfer rate incentive · branch consult |

Each diagnosed customer is paired with an **LLM-generated retention email** matched to their archetype, shown live in the dashboard.

> **한글 정리** — 고위험 고객은 이탈 요인이 서로 달라 처방도 다릅니다. 상품과다형·중장년형·독일형·비활성형으로 진단하고 유형별 리텐션 전략과 LLM 맞춤 이메일을 대시보드에서 제시합니다.

---

## Key Findings

- **Product cliff** — churn jumps from ~6% (2 products) to **88% (3+ products)**: the strongest single signal.
- **Balance paradox = geography confounding** — within Germany, balance is uncorrelated with churn (r ≈ 0.005); the real driver is *being in Germany*, not the balance itself. Correlation ≠ causation.
- **Irrelevant features** — credit score, tenure, salary, and card ownership show little effect and are excluded from retention budgeting.

> **한글 정리** — 상품 2개(6%) ↔ 3개↑(88%)의 '상품 절벽'이 최강 신호입니다. '잔고 역설'은 국가(독일) 교란 효과이며(독일 내 상관 ≈ 0.005), 신용점수·거래기간·연봉·카드보유는 이탈과 무관해 리텐션 예산에서 제외했습니다.

---

## Repository Structure

```
2026-sinchon-hackathon-team16/
├── index.html    # Interactive retention dashboard (self-contained, no dependencies)
├── report.html   # XAI deep-dive analysis report (self-contained)
└── README.md
```

Both pages are **fully self-contained** HTML (inline data, CSS, and JS) with no external dependencies, served via GitHub Pages. The full submission package — analysis notebooks, model outputs, and presentation deck — is maintained with the competition submission.

> **한글 정리** — 이 저장소는 GitHub Pages로 배포되는 두 개의 자체 완결형 HTML(대시보드·리포트)로 구성됩니다. 전체 분석 노트북·모델 산출물·발표자료는 해커톤 제출물로 별도 관리됩니다.

---

## Reproducibility & Tech

- **Model reproducibility** — all metrics, counterfactual, and ROI figures are reproduced by the submitted analysis code (5-fold CV, `FAST=False` for scoring).
- **Compliance** — `id` excluded from features · no leakage from the evaluation set · probability submission.
- **Stack** — scikit-learn · CatBoost · LightGBM · XGBoost · SHAP · pandas · numpy · matplotlib.

> **한글 정리** — 모든 수치는 제출 분석 코드로 재현 가능합니다(5-fold CV). id 미사용·평가셋 누수 없음·확률 제출 등 규정을 준수했으며, 사용 스택은 scikit-learn·CatBoost·LightGBM·XGBoost·SHAP입니다.

---

## Team & Competition

**Team 16** — 2026 Sinchon Union Data·AI Hackathon.
Model: CatBoost · LightGBM · XGBoost ensemble · Explainability: SHAP · Counterfactual/ROI reproduced by submitted code.

> **한글 정리** — 2026 신촌연합 데이터·AI 해커톤 16조 제출 프로젝트입니다.
