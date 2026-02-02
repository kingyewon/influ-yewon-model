# 인플루엔자 예측 모델 (PatchTST)

**시계열 데이터 기반 인플루엔자 유사질환(ILI) 발생률 예측 모델**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## 📋 프로젝트 개요

본 프로젝트는 **딥러닝 기반 PatchTST(Patch Time Series Transformer)** 모델을 활용하여 인플루엔자 유사질환(ILI) 발생률을 예측하는 시스템입니다.

### 주요 특징

- ✅ **예측 대상**: 주간 인플루엔자 유사질환(ILI) 발생률
- ✅ **예측 범위**: 향후 3주
- ✅ **데이터 기간**: 2015년 ~ 2025년 (주간 데이터)
- ✅ **주기성 특징 자동 생성**: week_sin, week_cos (52주 주기)
- ✅ **CSV 기반 데이터 처리**: pandas를 통한 데이터 로딩 및 병합
- ✅ **완전 자동화된 파이프라인**: 데이터 병합부터 예측까지 자동화

---

## 📁 프로젝트 구조

```
patchTST-yewon-demo/
├── main.py                      # 메인 실행 스크립트
├── config.py                    # 설정 상수 (하이퍼파라미터, 경로 등)
├── utils.py                     # 유틸리티 함수들
├── data_loader.py               # 데이터 로딩 및 전처리
├── model.py                     # 모델 클래스들 (PatchTSTModel 등)
├── train.py                     # 학습 및 평가 함수
├── feature_importance.py        # Feature Importance 계산
│
├── patchTST.ipynb              # 메인 모델 학습 노트북 (참고용)
├── MergeDATA.ipynb             # 데이터 병합 노트북
├── README.md                   # 프로젝트 개요 (본 파일)
├── PROJECT_REPORT.md           # 상세 개발 보고서
├── requirements.txt            # Python 의존성
│
├── data/                       # 데이터 파일
│   ├── raw/                    # 원본 데이터 CSV
│   │   ├── influenza_ili.csv
│   │   ├── vaccine.csv
│   │   ├── respiratory.csv
│   │   ├── weather.csv
│   │   └── merge_data.csv
│   ├── processed/              # 처리된 데이터 CSV
│   │   ├── 3_merged_influenza_vaccine_respiratory_weather.csv
│   │   └── 3_merged_influenza_vaccine_respiratory_weather_filled.csv
│   └── results/                # 예측 결과 및 시각화
│       ├── ili_predictions.csv
│       ├── feature_importance.csv
│       └── ...
│
├── patchtst_model.pth          # 학습된 모델 가중치 (선택사항)
├── patchtst_scalers.pkl        # 스케일러 저장 (선택사항)
└── plot_*.png                  # 시각화 결과 파일들
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 병합 (MergeDATA.ipynb 실행)
# data/raw/ 폴더에 원본 CSV 파일들이 있어야 합니다:
# - influenza_ili.csv
# - vaccine.csv
# - respiratory.csv
# - weather.csv
```

### 3. 모델 학습 및 예측

**방법 1: Python 스크립트 실행 (권장)**

```bash
# 메인 스크립트 실행
python main.py
```

**방법 2: Jupyter Notebook 실행**

```bash
# patchTST.ipynb 노트북 실행
jupyter notebook patchTST.ipynb

# 노트북의 모든 셀을 순차적으로 실행하면 자동으로 학습 및 평가가 수행됩니다.
```

---

## 📊 모델 성능

| 지표 | 값 |
|------|-----|
| **Test MAE** | **3.44** |
| Test MSE | 52.41 |
| Test RMSE | 7.24 |

*성능 지표는 실제 실행 결과입니다.*

---

## 🏗️ 모델 아키텍처

### PatchTST 구조

```
입력 데이터 (12주 × 특징 수)
    ↓
패치 분할 (3개 패치 × 4주)
    ↓
멀티스케일 CNN 임베딩 (4개 분기 병렬)
    ↓
토큰 컨볼루션 믹서
    ↓
위치 인코딩
    ↓
Transformer Encoder (4 layers, 2 heads)
    ↓
어텐션 풀링
    ↓
예측 헤드 (MLP: 128 → 64 → 64 → 3)
    ↓
출력 (향후 3주 ILI 예측값)
```

### 주요 구성 요소

- **멀티스케일 CNN**: 다양한 시간 스케일 패턴 포착 (커널 크기: 1, 3, 5, dilation=2)
- **Transformer Encoder**: 패치 간 장기 의존성 학습 (4 layers, 2 heads)
- **어텐션 풀링**: 중요한 패치에 가중치 부여
- **학습 설정**: Huber Loss, AdamW optimizer, Cosine Annealing LR scheduler with warmup

---

## 🔧 주요 기능

### 1. 데이터 병합 및 관리

- **CSV 파일 기반**: pandas를 통한 데이터 로딩 및 병합
- **다중 데이터셋 통합**: ILI, 백신, 호흡기, 기후 데이터 병합
- **시즌/주차 변환**: 시즌 주차를 캘린더 주차로 자동 변환

### 2. 데이터 전처리

- **주간 → 일간 보간**: 선형 보간을 통한 일간 데이터 생성
- **주기성 특징 생성**: week_sin, week_cos (52주 주기)
- **RobustScaler 정규화**: 이상치에 강건한 스케일링
- **결측치 처리**: 선형 보간 및 median 채우기

### 3. 모델 학습

- **자동화된 학습 파이프라인**: Train/Val/Test 자동 분할 (70%/15%/15%)
- **Early Stopping**: 과적합 방지 (patience=60)
- **손실 함수**: Huber Loss (delta=1.0)
- **학습률 스케줄링**: Cosine Annealing with Warmup (30 epochs)
- **상관계수 모니터링**: 검증 세트에서 상관계수 추적

### 4. 예측 및 평가

- **자동 예측 생성**: 테스트 데이터 기반 예측
- **성능 평가**: MAE, MSE, RMSE 지표
- **Feature Importance 분석**: Perturbation 기반 중요도 계산
- **시각화 자동 생성**: 예측 결과 그래프 생성 (3종)

---

## 📈 데이터 구성

### 원본 데이터 파일

| 파일명 | 설명 | 주요 컬럼 |
|--------|------|----------|
| `influenza_ili.csv` | 의사환자 분율 | 시즌, 주차, ILI 발생률 |
| `vaccine.csv` | 예방접종률 | 시즌, 예방접종률 |
| `respiratory.csv` | 호흡기 질환 데이터 | 시즌, 주차, 호흡기 감염 지수 |
| `weather.csv` | 기후 데이터 | 연도, 주차, 온도, 습도, 강수량 |

### 병합된 데이터

- **MergeDATA.ipynb** 실행 시 `data/processed/` 폴더에 병합된 CSV 파일 생성
- **주요 특징**: ILI, vaccine_rate, respiratory_index, 기후 변수들
- **주기성 특징**: week_sin, week_cos 자동 생성

---

## 🛠️ 기술 스택

### Python 패키지

- **딥러닝**: PyTorch
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn
- **시각화**: matplotlib, seaborn

---

## 📝 사용 예시

### Python 스크립트 실행 (권장)

```bash
# 메인 스크립트 실행
python main.py
```

실행하면 자동으로:
1. 데이터 로드 및 전처리
2. 모델 학습 (Huber Loss, AdamW optimizer)
3. 성능 평가 및 시각화 생성
4. Feature Importance 계산 및 저장
5. 예측 결과 저장

### 개별 모듈 사용

```python
import config
import data_loader
import train
import feature_importance

# 데이터 로드
X, y, labels, feat_names = data_loader.load_and_prepare(
    config.CSV_PATH, 
    use_exog=config.USE_EXOG
)

# 모델 학습 (Feature Importance 포함)
model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train.train_and_eval(
    X, y, labels, feat_names,
    compute_fi=True,  # Feature Importance 계산
    save_fi=True      # 결과 저장
)

# Feature Importance만 별도로 계산
fi_df = feature_importance.compute_feature_importance(
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc,
    scaler_y=scaler_y, feat_names=feat_names
)
```

### Jupyter Notebook 실행

```bash
# patchTST.ipynb 노트북 실행
jupyter notebook patchTST.ipynb
```

### 주요 하이퍼파라미터

하이퍼파라미터는 `config.py`에서 설정할 수 있습니다:

```python
SEQ_LEN = 12        # 입력 시퀀스 길이 (12주)
PRED_LEN = 3        # 예측 길이 (3주)
PATCH_LEN = 4       # 패치 크기 (4주)
D_MODEL = 128       # 임베딩 차원
N_HEADS = 2         # 어텐션 헤드 수
ENC_LAYERS = 4      # Transformer 레이어 수
BATCH_SIZE = 64     # 배치 크기
LR = 5e-4           # 학습률
EPOCHS = 100        # 최대 에포크
```

### 모델 구조

모델은 `model.py`에 정의되어 있으며, `train.py`의 `train_and_eval()` 함수를 통해 학습 및 평가가 수행됩니다.

---

## 📊 Feature Importance

Perturbation 기반 Feature Importance 분석을 통해 각 특징의 중요도를 계산합니다.

주요 특징:
- **ili (의사환자 분율)**: 과거 ILI 값이 가장 중요한 예측 변수
- **week_sin, week_cos**: 주기성 특징이 계절성 패턴을 잘 포착
- **vaccine_rate (예방접종률)**: 백신 접종률이 발생률에 영향
- **respiratory_index**: 호흡기 감염 지수
- **기후 변수들**: 온도, 습도, 강수량 등

*실제 중요도 순위는 `data/results/feature_importance.csv` 파일에서 확인할 수 있습니다.*

---

## 🔄 자동화 파이프라인

모든 과정이 **수동 개입 없이 재현 가능**한 구조로 설계되었습니다:

1. ✅ 데이터 로딩 및 전처리 (`data_loader.py`)
2. ✅ Train/Val/Test 분할 (70%/15%/15%)
3. ✅ 모델 학습 (Early stopping 포함) (`train.py`)
4. ✅ 성능 평가 (MAE, MSE, RMSE)
5. ✅ Feature Importance 계산 (`feature_importance.py`)
6. ✅ 시각화 생성 및 저장
7. ✅ 예측 결과 CSV 저장

## 📦 모듈 구조

프로젝트는 다음과 같이 모듈화되어 있습니다:

- **`config.py`**: 모든 설정 상수 (하이퍼파라미터, 경로, 디바이스 등)
- **`utils.py`**: 유틸리티 함수 (시드 설정, CSV 읽기, 스케일러 생성, 주간→일간 보간 등)
- **`data_loader.py`**: 데이터 로딩 및 전처리 (`load_and_prepare`)
- **`model.py`**: 모델 클래스들 (`PatchTSTModel`, `PatchTSTDataset` 등)
- **`train.py`**: 학습 및 평가 함수 (`train_and_eval`)
- **`feature_importance.py`**: Feature Importance 계산 및 저장
- **`main.py`**: 메인 실행 스크립트

---

## 📚 문서

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: 상세 개발 진행 보고서

---

**최종 업데이트**: 2026년 1월 29일

**문서 버전**: 1.1

**변경 사항**:
- 노트북 코드를 Python 모듈로 분리 (config.py, utils.py, data_loader.py, model.py, train.py, feature_importance.py, main.py)
- Python 스크립트 실행 방법 추가