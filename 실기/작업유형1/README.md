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

```python
ratings = (df["rating"] >=3) #-- 평점이 3이상만 필터링
useID = (df["userId"] <=5) #-- Id가 5이히만 필터링
print(len(df[ratings]))
print(len(df[useID]))

df[ratings & useID] #-- 2개 조건을 필터링
```

# 4. 데이터 정렬

### 내림차순 정렬

```python
df = pd.read_csv(rating_path)
df.sort_values('rating', ascending = False).head(5)
```

### 오름차순 정렬

```python
df = pd.read_csv(rating_path)
df.sort_values('rating', ascending = True).head(5)
```

# 5. 데이터 변경

```python
df = pd.read_csv(rating_path)

#-- rating의 값이 3이하인 값은 0으로 만들고 나머지는 그냥 유지
df['rating'] = np.where(df['rating'] >= 3, df['rating'],0)
#-- rating 프레임의 데이터 중 조건식이 true이면 1번째, false이면 2번째 값 대입

df.sort_values('rating', ascending = True).head(5)
```

# 6. 데이터 결측치, 이상치, 중복치 제거(타이타닉 데이터세트)

```python
#-- 타이타닉 데이터 불러오기
import seaborn as sns
df = sns.load_dataset('titanic')
print(df.shape) #-- 15개의 변수와 891개의 변수
print(df.info())
df.head(5)

#-- 종속 변수 Y : 생존 여부(servied)
#-- 독립 변수 X : 성별, 나이등의 탑승자 정보
```

## 1. 결측치 처리

```python
print(df.isnull().sum()) #-- 결측치 확인
print(df.dropna(axis=0).shape) #-- 행 기준 제거 --> 데이터를 삭제함 (기본값이므로 axis값을 안줘도 된다.)
print(df.dropna(axis=1).shape) #-- 열 기준 제거 --> 변수를 삭제함
```

```python
df1 = df.copy()
df1 = pd.DataFrame(df)
#-- 데이터를 복사할 때 Seanborn 데이터를 다시 데이터프레임 해줘야 함
```

### 결측치 대체

```python
df1_median = df1['age'].median()
print(df1_median) #-- 중앙값 계산
 
df1_mean = df1['age'].mean()
print(df1_mean)   #-- 평균값 계산

df1['age'] = df['age'].fillna(df1_median) #-- 구한 중앙값(평균값)으로 결측치 대체
df1.isnull().sum()
```

## 2. 이상치 처리

### 상자그림 활용 (이상치 : Q1, Q3으로부터 1.5 * IQR을 초과하는 값)

```python
df = sns.load_dataset('titanic')
# df.head(5)
sns.boxplot(df['age'])
```

```python
#-- Q1, Q3, IQR(Q3-Q1)을 구하기
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
print("Q1 is : {} Q3 is {} IQR is {}".format(Q1, Q3, IQR))

#-- 위아래 이상치 구하기
upper = Q3 + (1.5*IQR)
lower = Q1 - (1.5*IQR)

print("upper is {} lower is {}".format(upper, lower))
```

```python
cond1 = (df['age'] <= upper)
cond2 = (df['age'] >= lower)
print(len(df[cond1 & cond2]))
```

### 표준정규분포를 활용(이상치 : +=3Z값을 넘어가는 값)

Z = (개별값 - 평균값) / 표준편차

```python
#-- 평균값, 표준편차 계산
mean_value = df['age'].mean()
std_value = df['age'].std()
print("mean_value is {},  std_value is {}".format(mean_value, std_value))

zscore = (df['age']-mean_value) / std_value
print("Z value is {}".format(zscore))

cond1 = (zscore > 3)
cond2 = (zscore < -3)
print(len(df[cond1]) + len(df[cond2]))
```

## 3. 중복값 처리

```python
df = sns.load_dataset('titanic')
print(df.shape)

df1 = df.copy()
df1 = df1.drop_duplicates() #-- 데이터가 중복이 있는경우 삭제

print(df1.shape)
```

# 데이터 스케일링(데이터 표준화, 정규화)

### 1. 데이터 표준화 (Z-score nomalization)

```python
from sklearn.preprocessing import StandardScaler
```

```python
df = sns.load_dataset('titanic')
df = pd.DataFrame(df)

zscaler = StandardScaler()
df['fare'] = zscaler.fit_transform(df[['fare']])
# df.head()

#-- 평균이 0 표준편차가 1인 정규분포로 변환이 잘 되었는지 확인
print("평균 값 {} 표준편차 값 {}".format(df['fare'].mean(), df['fare'].std()))
```

### 2. 데이터 정규화(min-max nomalization)

```python
from sklearn.preprocessing import MinMaxScaler
```

```python
df = sns.load_dataset('titanic')
df = pd.DataFrame(df)

min_max_scaler = MinMaxScaler()
df['fare'] = min_max_scaler.fit_transform(df[['fare']])

print("최소 값 {} 최대 값 {}".format(df['fare'].min(), df['fare'].max()))
```

# 데이터 합치기(나온적은 없음)

```python
df = sns.load_dataset('iris')
# df.head()

df1 = df.loc[0:30,]
df2 = df.loc[31:60,]

df_sum = pd.concat([df1, df2], axis = 0) #-- 행 방향으로 합침
print(df1.shape, df2.shape, df_sum.shape)
```

# 날짜/시간 데이터, 인덱스 다루기

### 1. 날짜 데이터 다루기

```python
#-- 임의의 날짜 데이터 생성
df = pd.DataFrame({
    '날짜':['20230105', '20230105', '20230223', '20230223', '20230312', '20230422', '20230511'],
    '물품':['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    '판매수' :[5, 10, 15, 15, 20, 25, 40],
    '개당수익' : [500, 600, 500, 600, 600, 700, 600]
  })

df.info()
```

```python
#-- 날짜 데이터를 데이터타입을 변경
df['날짜'] =  pd.to_datetime(df['날짜'])
#-- 년, 월, 일 데이터로 분할
df['year'] =  df['날짜'].dt.year
df['month'] =  df['날짜'].dt.month
df['day'] =  df['날짜'].dt.day
# df.head(5)

#-- 날짜 구간 필터링(방법 1)
df[df['날짜'].between('2023-01-01', '2023-01-31')]
```

```python
#-- 날짜 구간 필터링(방법 2)
#-- 임의의 날짜 데이터 생성
df = pd.DataFrame({
    '날짜':['20230105', '20230105', '20230223', '20230223', '20230312', '20230422', '20230511'],
    '물품':['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    '판매수' :[5, 10, 15, 15, 20, 25, 40],
    '개당수익' : [500, 600, 500, 600, 600, 700, 600]
  })

df['날짜'] =  pd.to_datetime(df['날짜'])
df = df.set_index('날짜', drop = True) #-- Drop = True(디폴트) 해당 변수를 제거
# df
print(df.loc["2023-01-05": "2023-01-31"])
```

### 2. 시간 데이터 다루기

```python
#-- 임의의 시간 데이터 생성
df = pd.DataFrame({
    '물품':['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    '판매수' :[5, 10, 15, 15, 20, 25, 40],
    '개당수익' : [500, 600, 500, 600, 600, 700, 600]
  })

time = pd.date_range('2023-09-24 12:25:00' ,'2023-09-25 14:45:30', periods = 7)
df['time'] = time
df = df [['time', '물품', '판매수', '개당수익']] #-- 위치 재정렬

df = df.set_index('time') #-- time으로 인덱스 설정
```

```python
df.loc["2023-09-24 12:25:00":"2023-09-25 00:25:00"]
```