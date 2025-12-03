# Kiln Classification Pipeline 

This repository contains the complete pipeline for classifying **zigzag kilns vs. fixed chimney kilns (FCK)** in Bangladesh using **satellite imagery**.  
The project was part of SFU STAT 440 and focuses on building a high-performance machine learning pipeline under competition-style rules.

---

## 1. Project Overview

According to international environmental reports, brick kilns are a top contributor to air pollution in Bangladesh.  
The government requires replacing traditional kilns with modern zigzag kilns—so **detecting kiln types from satellite images** is essential.

### **Task**
- Binary image classification  
- `1` = Zigzag kiln (modern, less polluting)  
- `0` = FCK kiln (traditional, high pollution)

### **Dataset**
- **1,617** labeled training images (256×256 PNGs)  
- **724** test images  
- Evaluation metric: **AUC (ROC curve)**  
- Additional competition scoring components (RMSE, MAE), but AUC was dominant

---

## 2. Technical Summary

This project uses **EfficientNet-based models** with:
- Training on subset and full datasets  
- 5-fold cross-validation  
- Convolutional neural networks  
- Test-time augmentation (TTA)  
- Weighted model ensembling  
- PyTorch / Python

The final submission is an **ensemble** of:
1. Model 41 (subset-trained — best single model)  
2. 5-fold CV (subset)  
3. 5-fold CV (full data, including model 43)  
4. Model 43 (full-data model)

> This ensemble achieved our strongest performance across both public and private test splits.

---

## 3. Repository Structure
```
kiln-cv-pipeline/
|
|-- src/                      # Training & ensemble source code
|   |-- submission3_97851_41_cleaned.py        # Model 41 (subset)
|   |-- submission3_97851_43_cleaned.py        # Model 43 (full)
|   |-- submission3_97851_411_5cv_cleaned.py   # 5-fold CV
|   `-- finalfinalensemble_cleaned.py          # Final ensemble logic
|
|-- outputs/                  # Prediction CSV files
|   |-- submission411_5cv_s.csv                # 5CV (subset)
|   |-- submission411_5cv_f.csv                # 5CV (full)
|   |-- submission4_efficientnet1_99686.csv    # EfficientNet single model
|   `-- finalensemble_var1_top3.csv            # Final ensemble output
|
`-- README.md
```

---

## 4. Model Versions Explained

### ** Model 41 — Subset EfficientNet**
- Trained on curated training set (20% removed)  
- Best single-model AUC  
- Used in ensemble

---

### ** Model 43 — Full-Data EfficientNet**
- Trained on 100% of the training data  
- Slightly weaker than 41 (more noisy labels)  
- Provides variance → improves ensemble performance

---

### ** 5-Fold Cross-Validation Models**
Two types were generated:

| Version | Description |
|--------|-------------|
| `_s` | Trained on subset-only folds |
| `_f` | Includes full-model predictions (mix of model 43) |

These improve robustness across train/test distribution shifts.

---

### ** Final Ensemble (`finalensemble_var1_top3.csv`)**
The winning blend includes:
- 5× subset CV models  
- 5× full-data CV models  
- Model 41  
- Model 43  

Weighted averaging was used based on validation AUC.

---

## 5. Tools & Libraries Used

- Python 3  
- PyTorch  
- EfficientNet B0–B4  
- NumPy  
- Pandas  
- scikit-learn  
- OpenCV  
- tqdm  
- CUDA (for training)

---

## 6. Data Handling Notes

Due to academic rules:
- Training/test images **cannot be uploaded**
- Place them locally like:
data/train/
data/test/

Each file name corresponds to index:
KXXXX.png → index = KXXXX

---

## 7. How to Run (local)
python src/submission3_97851_41_cleaned.py
python src/submission3_97851_43_cleaned.py
python src/submission3_97851_411_5cv_cleaned.py
python src/finalfinalensemble_cleaned.py

The ensemble script will generate:
outputs/finalensemble_var1_top3.csv

---

## Author

**Dahyeon Choi (Data Science)**  
GitHub: https://github.com/dahyun0225

---

---

# Kiln Classification Pipeline (STAT 440 Project 2)

이 저장소는 **위성 이미지 기반 벽돌가마 분류 프로젝트**(지그재그 vs 전통식 FCK)를 위한  
완전한 머신러닝 파이프라인을 포함합니다.

실제 대회 형식의 평가(AUC 중심)를 따릅니다.

---

## 1. 프로젝트 개요

방글라데시는 전통식 벽돌가마(FCK)가 주요 대기오염원 중 하나입니다.  
정부는 오염이 적은 **Zigzag Kiln**으로 전환을 요구하고 있으며,  
이를 위해 위성사진으로 가마 종류를 자동 분류하는 것이 필요합니다.

### **목표**
- 이진 분류  
- `1` = Zigzag kiln  
- `0` = FCK kiln  

### **데이터**
- 1,617개 학습 이미지  
- 724개 테스트 이미지  
- 주요 평가지표: **AUC(ROC)**

---

## 2. 기술 요약

본 프로젝트는 EfficientNet 기반 모델을 사용하며:

- 부분 데이터(subset) & 전체 데이터(full) 훈련
- 5폴드 크로스밸리데이션
- PyTorch 기반 CNN
- Test-Time Augmentation (TTA)
- 다중 모델 앙상블

최종 제출 파일은 다음 모델들의 **가중 앙상블**입니다:

1. Model 41 (subset — 최고 단일 모델)
2. subset 기반 5CV  
3. full-data 기반 5CV  
4. Model 43 (full model)

---

## 3. 저장소 구조

```
kiln-cv-pipeline/
|
|-- src/                      # Training & ensemble source code
|   |-- submission3_97851_41_cleaned.py        # Model 41 (subset)
|   |-- submission3_97851_43_cleaned.py        # Model 43 (full)
|   |-- submission3_97851_411_5cv_cleaned.py   # 5-fold CV
|   `-- finalfinalensemble_cleaned.py          # Final ensemble logic
|
|-- outputs/                  # Prediction CSV files
|   |-- submission411_5cv_s.csv                # 5CV (subset)
|   |-- submission411_5cv_f.csv                # 5CV (full)
|   |-- submission4_efficientnet1_99686.csv    # EfficientNet single model
|   `-- finalensemble_var1_top3.csv            # Final ensemble output
|
`-- README.md
```

---

## 4. 모델 버전 설명

### Model 41 — Subset 기반 EfficientNet  
- 노이즈가 적은 부분 데이터로 학습  
- 최고 단일 모델 AUC 기록  

### Model 43 — Full-data EfficientNet  
- 전체 데이터 기반  
- 단일 모델은 약하지만 앙상블에 기여도 높음  

### 5-Fold Cross Validation  
- `_s`: subset CV  
- `_f`: full+subset hybrid CV  

### Final Ensemble  
총 12개 모델 조합:  
- 5CV(subset) ×5  
- 5CV(full) ×5  
- Model 41  
- Model 43  
→ best weighted ensemble: `finalensemble_var1_top3.csv`

---

## 5. 사용 라이브러리

Python / PyTorch / EfficientNet / NumPy / Pandas / scikit-learn / OpenCV / CUDA / tqdm

---

## 6. 데이터 구조

대회 규정상 이미지 업로드 불가.  
로컬에서:
data/train/
data/test/

---

## 7. 실행 방법
python src/submission3_97851_41_cleaned.py
python src/submission3_97851_43_cleaned.py
python src/submission3_97851_411_5cv_cleaned.py
python src/finalfinalensemble_cleaned.py

출력 위치:
outputs/finalensemble_var1_top3.csv

---

## 작성자  
**최다현 (Dahyeon Choi)**  
GitHub: https://github.com/dahyun0225






