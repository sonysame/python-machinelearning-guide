# 데이터 전처리

## 데이터 인코딩

머신러닝을 위한 대표적인 인코딩 방식은 **레이블 인코딩**과 **원-핫 인코딩**이 있다. 

### 레이블 인코딩


```python
from sklearn.preprocessing import LabelEncoder

items=['TV', '냉장고','전자레인지','컴퓨터','선풍기','믹서','믹서']

encoder=LabelEncoder()
encoder.fit(items)
labels=encoder.transform(items)
print('인코딩 변환값:', labels)
```

    인코딩 변환값: [0 1 4 5 3 2 2]
    


```python
encoder.classes_
```




    array(['TV', '냉장고', '믹서', '선풍기', '전자레인지', '컴퓨터'], dtype='<U5')




```python
encoder.inverse_transform([4,5,2,0,1,1,3,3])
```




    array(['전자레인지', '컴퓨터', '믹서', 'TV', '냉장고', '냉장고', '선풍기', '선풍기'],
          dtype='<U5')



### 원-핫 인코딩


```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items=['TV', '냉장고','전자레인지','컴퓨터','선풍기','믹서','믹서']
encoder=LabelEncoder()
encoder.fit(items)
labels=encoder.transform(items)

labels=labels.reshape(-1,1)
oh_encoder=OneHotEncoder()
oh_encoder.fit(labels)
oh_labels=oh_encoder.transform(labels)
print(oh_labels.shape)
oh_labels.toarray()
```

    (7, 6)
    




    array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.]])



pandas에는 원-핫 인코딩을 더 쉽게 지원하는 API인 **get_dummies()** 가 있다


```python
import pandas as pd
df=pd.DataFrame({'item':['TV', '냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_TV</th>
      <th>item_냉장고</th>
      <th>item_믹서</th>
      <th>item_선풍기</th>
      <th>item_전자레인지</th>
      <th>item_컴퓨터</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 피처 스케일링과 정규화

서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업을 **피처 스케일링**이라고 한다.<br/>
대표적인 방법으로 **표준화(Standardization)**과 **정규화(Normalization)**이 있다.

표준화: 데이터의 피처 각각이 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환<br/>
x에서 평균을 뺀 값을 표준편차로 나눈 값!

정규화: 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해주는 개념

사이킷런에서 제공하는 대표적인 피처 스케일링 -> **StandardScaler**, **MinMaxScaler**<br/>
데이터의 스케일링 변환 시, **fit()**,**transform()**,**fit_transform()** 메소드를 이용
* fit()은 데이터 변환을 위한 기준 정보 설정
* transform()은 설정된 정보를 이용해 데이터 변환
* fit_transform()은 fit()과 transform()을 한번에 적용

### StandardScaler : 표준화

사이킷런에서 구현한 RBF 커널을 이용하는 서포트벡터머신, 선형회귀, 로지스틱 회귀는 데이터가 가우시안 분포를 가지고 있다고 가정하고 구현됐기 때문에 사전에 표준화를 적용하는 것은 예측 성능 향상에 좋다


```python
from sklearn.datasets import load_iris
import pandas as pd

iris=load_iris()
iris_data=iris.data
iris_df=pd.DataFrame(data=iris_data, columns=iris.feature_names)
print("feature들의 평균값")
print(iris_df.mean())
print("\nfeature들의 분산 값")
print(iris_df.var())
```

    feature들의 평균값
    sepal length (cm)    5.843333
    sepal width (cm)     3.057333
    petal length (cm)    3.758000
    petal width (cm)     1.199333
    dtype: float64
    
    feature들의 분산 값
    sepal length (cm)    0.685694
    sepal width (cm)     0.189979
    petal length (cm)    3.116278
    petal width (cm)     0.581006
    dtype: float64
    


```python
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(iris_df)
iris_scaled=scaler.transform(iris_df)
#iris_scaled는 ndarray -> DataFrame으로 변환!
iris_df_scaled=pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print("feature들의 평균값")
print(iris_df_scaled.mean())
print("\nfeature들의 분산 값")
print(iris_df_scaled.var())
```

    feature들의 평균값
    sepal length (cm)   -1.690315e-15
    sepal width (cm)    -1.842970e-15
    petal length (cm)   -1.698641e-15
    petal width (cm)    -1.409243e-15
    dtype: float64
    
    feature들의 분산 값
    sepal length (cm)    1.006711
    sepal width (cm)     1.006711
    petal length (cm)    1.006711
    petal width (cm)     1.006711
    dtype: float64
    

### MinMaxScaler : 정규화

데이터값을 0과 1사이의 범위로 변환(음수 값이 있으면 -1에서 1값으로 변환)


```python
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(iris_df)
iris_scaled=scaler.transform(iris_df)

iris_df_scaled=pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('features들의 최솟값')
print(iris_df_scaled.min())
print("\nfeatures들의 최댓값")
print(iris_df_scaled.max())
```

    features들의 최솟값
    sepal length (cm)    0.0
    sepal width (cm)     0.0
    petal length (cm)    0.0
    petal width (cm)     0.0
    dtype: float64
    
    features들의 최댓값
    sepal length (cm)    1.0
    sepal width (cm)     1.0
    petal length (cm)    1.0
    petal width (cm)     1.0
    dtype: float64
    

### 학습 데이터와 테스트 데이터의 스케일링 변환 시 유의점

학습 데이터 세트로 fit()을 수행한 결과를 이용해 테스트 데이터 세트의 transform() 변환을 해야 학습 데이터와 테스트 데이터의 스케일링 기준 정보가 같게 된다. 


```python
import numpy as np

train_array=np.arange(0,11).reshape(-1,1)
test_array=np.arange(0,6).reshape(-1,1)

scaler=MinMaxScaler()
scaler.fit(train_array)
train_scaled=scaler.transform(train_array)

print('원본 train_array 데이터:', np.round(train_array.reshape(-1),2))
print('Scale된 train_array 데이터:',np.round(train_scaled.reshape(-1),2))
```

    원본 train_array 데이터: [ 0  1  2  3  4  5  6  7  8  9 10]
    Scale된 train_array 데이터: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    


```python
#scaler.fit(test_array)
test_scaled=scaler.transform(test_array)

print('원본 test_array 데이터:', np.round(test_array.reshape(-1),2))
print('Scale된 test_array 데이터:',np.round(test_scaled.reshape(-1),2))
```

    원본 test_array 데이터: [0 1 2 3 4 5]
    Scale된 test_array 데이터: [0.  0.1 0.2 0.3 0.4 0.5]
    
