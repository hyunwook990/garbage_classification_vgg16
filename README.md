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

### 2025-05-07
- 결국 만들어야할 것은
    - `train_data` : `img`
    - `train_label` : `img`들의 `label`(보통 `labelencoding`된 상태로 저장)
    - `test_data` : `test_img`
    - `train_classes` : `idx : 'label'`의 형태로 저장된 `dict`

### 문제점
- 이미지파일 이름에 `label`이 있는 형식이라 이미지파일에서 `.split`을 사용해 `label`을 추출하는 중에 이미지파일 이름이 이런 형식으로 되어있지 않아 문제가 발생
- 이 문제는 두개의 파일에서만 발생했기에 따로 `label`을 설정해주면서 해결함

- `data_save` 완료

### 2025-05-10
- 문제점
    - `csv`파일로 저장한 `train_data, test_data`의 `img`가 개행문자에 의해서 `str`로 저장이 되는 문제가 발생했다.
    - `.hdf5`로 저장해 모델을 돌려보려고 했으나 이미지의 크기가 제각각이라 저장이 되지않음
- 해결방법
    - `cv2.resize`로 해결하여 `.hdf5`파일로 저장

### 2025-05-12
- kaggle에 제출하는 파일은 `.csv`파일이라고해서 `image`를 `.csv`파일로 저장해야하는데 이런 경우에는 단순히 `image`자체를 저장하기보단 경로를 저장하는 방식으로 진행해야한다.

- 문제점
    - `train, test`, 실행 함수가 잘못되어서 수정을 해야하는 문제가 발생했다.
    - 아직 공부중이라 코드를 함부로 고칠 수 없기에 일단은 잠정 휴식하고 논문을 다시 시작하겠다.
