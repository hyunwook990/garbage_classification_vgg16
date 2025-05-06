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
1. test, train, val 폴더 경로 구분
2. os.listdir로 label을 저장
3. 각 label의 갯수만큼 label 갯수를 늘려서 image, label을 매칭해 csv파일로 따로 저장
4. val dataset을 train dataset과 합쳐서 다양한 학습이 가능하게 함
