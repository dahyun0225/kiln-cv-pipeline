# Kiln Classification Pipeline

This repository contains the complete pipeline for classifying zigzag kilns vs. fixed chimney kilns (FCK) in Bangladesh using satellite imagery.  
The project builds a high-performance machine learning workflow designed for a competitive evaluation environment focused on AUC.

---

## 1. Project Overview

Brick kilns are a major contributor to air pollution in Bangladesh.  
Modern zigzag kilns emit less pollution, while traditional fixed chimney kilns (FCK) emit significantly more.  
Automated identification of kiln types from satellite images supports environmental monitoring and policy compliance.

### Task
- Binary image classification  
- `1` = Zigzag kiln  
- `0` = FCK kiln  

### Dataset
- 1,617 labeled training images (256×256 PNGs)  
- 724 test images  
- Primary evaluation metric: AUC (ROC curve)

---

## 2. Technical Summary

This project uses EfficientNet-based convolutional neural networks with:

- Training on both subset and full datasets
- 5-fold cross-validation
- Test-time augmentation (TTA)
- Weighted model ensembling
- PyTorch-based implementation

The final prediction file is an ensemble of:
1. Model 41 (subset-trained, highest-performing single model)  
2. Subset 5-fold CV  
3. Full-data 5-fold CV  
4. Model 43 (full-data model)  

The combined ensemble achieved strong performance across both public and private evaluation sets.

---

## 3. Repository Structure

```
kiln-cv-pipeline/
|
|-- src/                   
|   |-- submission3_97851_41_cleaned.py        
|   |-- submission3_97851_43_cleaned.py        
|   |-- submission3_97851_411_5cv_cleaned.py   
|   `-- finalfinalensemble_cleaned.py          
|
|-- outputs/
|   |-- submission411_5cv_s.csv                
|   |-- submission411_5cv_f.csv                
|   |-- submission4_efficientnet1_99686.csv    
|   `-- finalensemble_var1_top3.csv            
|
`-- README.md
```

---

## 4. Model Versions Explained

### Model 41 — Subset EfficientNet
- Trained on curated subset with noisy samples removed  
- Best single-model AUC  
- Used in the final ensemble

### Model 43 — Full-Data EfficientNet
- Trained on the entire dataset  
- Slightly weaker due to label noise  
- Adds variance that improves ensemble performance

### 5-Fold Cross-Validation Models
Two types were generated:

| Version | Description |
|--------|-------------|
| `_s` | Trained on subset-only folds |
| `_f` | Includes mixed predictions using the full-data model (Model 43) |

These CV outputs help stabilize predictions under distribution shifts.

### Final Ensemble
The final blended file (`finalensemble_var1_top3.csv`) incorporates:
- 5 subset CV models  
- 5 full-data CV models  
- Model 41  
- Model 43  

Weighted averaging was applied based on validation AUC.

---

## 5. Tools & Libraries

- Python  
- PyTorch  
- EfficientNet  
- NumPy  
- Pandas  
- scikit-learn  
- OpenCV  
- tqdm  
- CUDA (for GPU training)

---

## 6. Data Handling Notes

Training and test images are not included due to restrictions.  
To run the pipeline locally, prepare the following structure:

```
data/
 ├── train/
 └── test/
```

Each image file follows the format:

```
KXXXX.png  →  index = KXXXX
```

---

## 7. Execution

```
python src/submission3_97851_41_cleaned.py
python src/submission3_97851_43_cleaned.py
python src/submission3_97851_411_5cv_cleaned.py
python src/finalfinalensemble_cleaned.py
```

The final ensemble output will be saved to:

```
outputs/finalensemble_var1_top3.csv
```

---

# Kiln Classification Pipeline (Korean Version)

이 저장소는 위성 이미지에서 벽돌가마 종류(지그재그 vs 전통식 FCK)를 분류하기 위한  
머신러닝 파이프라인 전체를 포함합니다.

---

## 1. 프로젝트 개요

전통식 벽돌가마(FCK)는 방글라데시 대기오염의 주요 원인으로 알려져 있습니다.  
반면 Zigzag Kiln은 상대적으로 오염이 적어 정책적으로 권장됩니다.  
이 프로젝트는 위성 사진만으로 가마 종류를 자동 분류하는 모델을 구축하는 데 목표가 있습니다.

### 분류 대상
- 1 = Zigzag Kiln  
- 0 = FCK  

### 데이터 구성
- 1,617개 학습 이미지  
- 724개 테스트 이미지  
- 평가 지표: AUC

---

## 2. 기술 요약

모델은 EfficientNet 기반 CNN으로 구성되며 다음 요소를 포함합니다:

- 부분 데이터(subset) 및 전체 데이터(full) 기반 모델 학습
- 5-Fold Cross-Validation
- Test-Time Augmentation
- 모델 가중 앙상블

최종 제출 파일은 다음 4개 모델을 결합한 앙상블입니다:
1. Model 41 (subset, 최고 단일 모델)  
2. subset 기반 5CV  
3. full-data 기반 5CV  
4. Model 43 (full model)  

---

## 3. 저장소 구조

```
kiln-cv-pipeline/
|
|-- src/
|   |-- submission3_97851_41_cleaned.py
|   |-- submission3_97851_43_cleaned.py
|   |-- submission3_97851_411_5cv_cleaned.py
|   `-- finalfinalensemble_cleaned.py
|
|-- outputs/
|   |-- submission411_5cv_s.csv
|   |-- submission411_5cv_f.csv
|   |-- submission4_efficientnet1_99686.csv
|   `-- finalensemble_var1_top3.csv
|
`-- README.md
```

---

## 4. 모델 설명

### Model 41  
- 노이즈 제거된 subset 기반  
- 가장 높은 단일 모델 성능  

### Model 43  
- 전체 데이터 기반  
- 단일 모델 성능은 낮지만 앙상블 기여도 높음  

### 5-Fold CV  
- `_s` : subset 기반  
- `_f` : full + subset 혼합 기반  

### Final Ensemble  
총 12개 모델을 결합하여 최종 성능 향상  
결과물: `finalensemble_var1_top3.csv`

---

## 5. 사용 라이브러리

Python / PyTorch / EfficientNet / NumPy / Pandas / scikit-learn / OpenCV / CUDA / tqdm

---

## 6. 데이터 구조

데이터는 규정상 공유할 수 없으며 아래와 같이 로컬에 구성해야 합니다:

```
data/
 ├── train/
 └── test/
```

---

## 7. 실행 방법

```
python src/submission3_97851_41_cleaned.py
python src/submission3_97851_43_cleaned.py
python src/submission3_97851_411_5cv_cleaned.py
python src/finalfinalensemble_cleaned.py
```

출력 파일은 다음 경로에 저장됩니다:

```
outputs/finalensemble_var1_top3.csv
```

---

작성자: Dahyeon Choi  
GitHub: https://github.com/dahyun0225
