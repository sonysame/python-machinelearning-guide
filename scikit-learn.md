 ## 붓꽃 품종 예측하기
 
* 붓꽃 데이터 세트 생성 -> load_iris()
* ML 알고리즘 -> 의사결정트리(DecisionTreeClassifier)
* 데이터세트를 학습 데이터와 테스트 데이터로 분리 -> train_test_split()


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=load_iris()
iris_data=iris.data
iris_label=iris.target

iris_df=pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
print(iris.target_names)
iris_df
```

    ['setosa' 'versicolor' 'virginica']
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



### train_test_split
test_size=0.2로 분류하면 전체 데이터 중 테스트 데이터가 20%, 학습 데이터가 80%로 분할

1. 데이터세트 분리
2. 모델학습: **fit()**
3. 예측수행: **predict()**
4. 평가: **accuracy_score()**


```python
X_train, X_test, y_train, y_test=train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

dt_clf=DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train,y_train)
pred=dt_clf.predict(X_test)

print('예측 정확도: {0:4f}'.format(accuracy_score(y_test,pred)))
```

    예측 정확도: 0.933333
    

사이킷런에서는 분류 알고리즘을 구현한 클래스를 **Classifier**로, 회귀 알고리즘을 구현한 클래스를 **Regressor** 클래스로 지칭

**Classifier**와 **Regressor**를 합쳐서 **Estimator** 클래스라고 부른다

**Estimator** 클래스에서 fit()과 predict()를 제공

**분류나 회귀 연습용 예제 데이터**

|API 명|설명|
|:---|:---:|
|datasets.load_boston()|회귀, 미국 보스턴의 집 피터들과 가격에 대한 데이터|
|datasets.load_breat_cancer()|분류, 위스콘신 유방암 피처들과 악성/음성 레이블 데이터|
|datasets.load_diabetes()|회귀, 당뇨 데이터|
|datasets.load_digits()|분류, 0에서 9까지 숫자의 이미지 픽셀 데이터|
|datasets.load_iris()|분류, 붓꽃에 대한 피처를 가진 데이터|
|fetch_convtype()|회귀 분석용 토지 조사 자료|
|fetch_20newsgroups()|뉴스 그룹 텍스트 자료|
|fetch_olivetti_faces()|얼굴 이미지 자료|
|fetch_lfw_people()|얼굴 이미지 자료|
|fetch_lfw_pairs()|얼굴 이미지 자료|
|fetch_rcv1()|로이터 뉴스 말뭉치|
|fetch_mldata()|ML 웹사이트에서 다운로드|
키는 보통 data, target(레이블 값), target_name, feature_names, DESCR로 구성

**분류와 클러스터링을 위한 표본 데이터 생성기**

* **datasets.make_classifications()** : 분류를 위한 데이터세트를 만든다. 특히 높은 상관도, 불필요한 속성 등의 노이즈 효과를 위한 데이터를 무작위로 생성
* **datasets.make_blobs()** : 클러스터링을 위한 데이터 세트를 무작위로 생성. 군집 지정 개수에 따라 여러 가지 클러스터링을 위한 데이터 세트를 쉽게 만들어준다.

### 교차검증

과적합(Overfitting): 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우에는 에측 성능이 과도하게 떨어지는 것


```python
from sklearn.model_selection import KFold

itis=load_iris()
features=iris.data
label=iris.target
dt_clf=DecisionTreeClassifier(random_state=156)

kfold=KFold(n_splits=5)
cv_accuracy=[]

n_iter=0
print(features.shape[0])
for train_index, test_index in kfold.split(features):
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=label[train_index], label[test_index]
    
    #학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred=dt_clf.predict(X_test)
    n_iter+=1
    
    accuracy=np.round(accuracy_score(y_test, pred),4)
    train_size=X_train.shape[0]
    test_size=X_test.shape[0]
    print("\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}".format(n_iter, accuracy, train_size, test_size))
    print("#{0} 검증 세트 인덱스:{1}".format(n_iter, test_index))
    cv_accuracy.append(accuracy)

print("\n## 평균 검증 정확도:", np.mean(cv_accuracy))
```

    150
    
    #1 교차 검증 정확도 :1.0, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #1 검증 세트 인덱스:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29]
    
    #2 교차 검증 정확도 :0.9667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #2 검증 세트 인덱스:[30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
     54 55 56 57 58 59]
    
    #3 교차 검증 정확도 :0.8667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #3 검증 세트 인덱스:[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
     84 85 86 87 88 89]
    
    #4 교차 검증 정확도 :0.9333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #4 검증 세트 인덱스:[ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119]
    
    #5 교차 검증 정확도 :0.7333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #5 검증 세트 인덱스:[120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149]
    
    ## 평균 검증 정확도: 0.9
    


