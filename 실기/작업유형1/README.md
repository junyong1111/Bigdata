# 작업유형 1

![1](https://github.com/junyong1111/Bigdata/assets/79856225/d5a0152f-0010-4e22-8555-5ffae4e6024e)


# 0. MovieLens 데이터를 활용

```bash
!pip install torch_geometric
```

```python
from torch_geometric.data import download_url, extract_zip
import pandas as pd
import numpy as np
#-- Libreco
# download the dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

rating_path = '/content/ml-latest-small/ratings.csv'
item_path = "/content/ml-latest-small/movies.csv"
```

```python
df = pd.read_csv(rating_path) #-- Ratings 데이터 읽어오기
print(df.dtypes)
df.head(5) #-- 최상단 5개의 프레임 확인
```

# 1. 데이터 확인

### 1-1. 데이터 타입 변경

```python
# df1 = df  #-- 이와 같이 복사하면 Call by value가 아닌 call by reference라 df가 변경되면 df1도 변경되므로 .copy()를 이용하자
df1= df.copy() #-- 복사
# df1 = df1.astype({'rating' : 'object'}) #-- 원하는 칼럼의 데이터 타입을 변경
df1 = df1.astype({'rating' : 'float64', 'userId' : 'float64'}) #-- 원하는 칼럼의 데이터 타입을 여러개 변경
print(df1.dtypes)
```

### 1-2. 원하는 데이터 출력

```python
print(df1['rating']) #-- userId 데이터만 가져옴

print(df1['rating'].value_counts()) #-- 각각의 값들이 몇 개 들어있는지 확인
```

# 2. 기초통계량(평균, 중앙값, IQR, 표준편차 등)

```python
df = pd.read_csv(rating_path)
print(df.shape) #-- 행렬 (100836, 4) -> 100836개의 데이터와 4개의 변수로 구성되어 있음
df.head(5)
```

### 평균값 구하기(mean)

```python
rating_mean = df['rating'].mean()
print(rating_mean)
```

### 중앙값 구하기(median)

```python
rating_median = df['rating'].median()
print(rating_median)
```

### 최빈값 구하기(mode)

```python
rating_mode = df['rating'].mode()
print(rating_mode[0])
# print(df['rating'].value_counts())
```

### 분산(var)

분산이 크면 데이터 값들이 평균에서 넓게 퍼져 있다는 것을 의미하고, 분산이 작으면 데이터 값들이 평균에 가까이 모여 있다는 것을 의미한다.

```python
rating_var = df['rating'].var()
print(rating_var)
```

### 표준편차(std)

표준편차가 0에 가깝다는 것은 모든 데이터 값들이 거의 또는 완전히 평균값에 근접해 있다는 것을 의미한다.
데이터 값들이 평균값에서 멀리 떨어져 있을수록 표준편차는 커진다.

```python
rating_std = df['rating'].std()
print(rating_std)
```

### IQR(Q3- Q1)

QR은 데이터의 중간 부분이 얼마나 넓게 퍼져 있는지를 나타냅니다.  
예를 들어, 어떤 시험의 점수가 10, 20, 50, 60, 70, 80, 90점으로 나타났다고 가정해봅시다. 이 경우 중앙값은 60점이고, 하위 50%의 중앙값(Q1)은 20점, 상위 50%의 중앙값(Q3)은 80점일 것입니다. 그러므로 IQR은 80 - 20 = 60점이 됩니다. 이것은 대부분의 학생들이 20점과 80점 사이에 점수를 가지고 있음을 의미하며, 이 범위 밖의 점수는 상대적으로 드물다는 것을 나타냅니다.

```python
Q3 = df['rating'].quantile(.75)
print("Q3 data is : {}".format(Q3))
Q1 = df['rating'].quantile(.25)
print("Q1 data is : {}".format(Q1))
IQR = Q3 - Q1
print("IQR data is : {}".format(IQR))
```

### 최대, 최소

```python
rating_max = df['rating'].max()
print("rating_max value is {}".format(rating_max))
rating_min = df['rating'].min()
print("rating_min value is {}".format(rating_min))
rating_range = rating_max - rating_min
print("rating_range is {}".format(rating_range))
```

### 왜도(skew)

```python
rating_skew = df['rating'].skew()
print(rating_skew)
```

### 첨도(kurt)

```python
rating_kurt = df['rating'].kurt()
print(rating_kurt)
```

# 3. 그룹화하여 데이터 전처리, 데이터 인덱싱, 필터링, 정렬

```python
import seaborn as sns
df = sns.load_dataset('iris')
print(df.head(5)) #-- species 재정렬

df.groupby('species').mean()
```

### 데이터 인덱싱

```python
df = pd.read_csv(rating_path)
print(df.shape) #-- 행렬 (100836, 4) -> 100836개의 데이터와 4개의 변수로 구성되어 있음
df.head(5)
df.loc[3, 'rating'] #-- 3번째가 아닌 인덱스번호 3을 가져옴
```

```python
df.loc[:, 'rating'] #-- rating만 가져옴
df.loc[0:3, ['rating', 'userId']] #-- 특정 인덱스구간의 원하는 데이터만 추출
```

```python
df.head(5)#-- 앞에서 5개
df.tail(5)#-- 뒤에서 5개
```

### 열 추가/제거

```python
df_rating = df.rating
# df_rating = df['rating']
df_rating.head(5)

#-- 여러개 가져오기
df1 = df[['rating', 'userId']]
df1.head(3)
```

```python
df_drop = df.copy()
df_drop = df_drop.drop(columns=['timestamp', 'movieId'])
df_drop.head(5)
```

```python
#-- 열 추가
df_new = df.copy()
df_new['new'] = df['rating'].mean()
df_new.head(3)
```

### 데이터 필터링