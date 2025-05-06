# VGG16을 활용한 garbage classification
- start_date: 2025-05-06
- dataset: kaggle https://www.kaggle.com/competitions/garbage-guru-challenge-2-0/data

## 데이터 로드
### 데이터 폴더 구성
```
dataset/
├── test/
│ ├── battery/
│ ├── biological/
│ ├── cardboard/
│ ├── clothes/
│ ├── glass/
│ ├── metal/
│ ├── paper/
│ ├── plastic/
│ ├── shoes/
│ └── trash/
├── train/
└── val/
```
- `train/`, `val/`, `test/` 폴더로 나뉘고 각 클래스별 하위 폴더로 구성됨.
- 이미지와 라벨 정보를 매칭하여 `.csv` 파일로 별도 저장함.

1. `os.listdir`을 통한 `label` 자동추출
2. `image`와 `label`을 매칭해 `dataframe` 생성
3. 학습 데이터 다양성을 높이기 위해 `val` 데이터를 `train`에 통합 사용

### 문제점
1. 경로 통일의 어려움
- `train`과 `val`의 경로가 다르기 때문에 병합후 작업시 경로 처리를 따로 해줘야 하는 번거로움이 있음.
2. 이미지 크기 불일치
- 데이터셋 내 이미지 크기가 통일되지 않아 모델 학습을 위해 `resize` 필요.
- 다만, `resize` 시점을 정확하게 알지 못함.