## Stratified K 폴드

-> 불균현한 분포도를 가진 레이블 데이터 집합을 위한 K 폴드 방식<br/>
K폴드가 레이블 데이터 집합이 원본 데이터 집합의 레이블 분포를 학습 및 테스트 세트에 제대로 분배하지 못하는 경우의 문제를 해결



```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
iris=load_iris()
iris_df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()

kfold=KFold(n_splits=3)
n_iter=0
for train_index, test_index in kfold.split(iris_df):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test=iris_df['label'].iloc[test_index]
    print("\n## 교차 검증: {0}".format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())
```

    
    ## 교차 검증: 1
    학습 레이블 데이터 분포:
     2    50
    1    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     0    50
    Name: label, dtype: int64
    
    ## 교차 검증: 2
    학습 레이블 데이터 분포:
     2    50
    0    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     1    50
    Name: label, dtype: int64
    
    ## 교차 검증: 3
    학습 레이블 데이터 분포:
     1    50
    0    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    50
    Name: label, dtype: int64
    

StrarifiedKFold는 split 메서드에 인자로 피처 데이터 세트와 레이블 데이터 세트가 있어야 한다. 


```python
from sklearn.model_selection import StratifiedKFold

skf=StratifiedKFold(n_splits=3)
n_iter=0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index] #label_train은 Series형태
    label_test=iris_df['label'].iloc[test_index]
    print("\n## 교차 검증: {0}".format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())
```

    
    ## 교차 검증: 1
    학습 레이블 데이터 분포:
     2    34
    1    33
    0    33
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     1    17
    0    17
    2    16
    Name: label, dtype: int64
    
    ## 교차 검증: 2
    학습 레이블 데이터 분포:
     1    34
    2    33
    0    33
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    17
    0    17
    1    16
    Name: label, dtype: int64
    
    ## 교차 검증: 3
    학습 레이블 데이터 분포:
     0    34
    2    33
    1    33
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    17
    1    17
    0    16
    Name: label, dtype: int64
    


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf=DecisionTreeClassifier(random_state=156)
skfold=StratifiedKFold(n_splits=3)
cv_accuracy=[]
n_iter=0

features=iris.data #iris_df.values(label이 포함되어 정확도 -> 1.0)
label=iris.target#iris_df['label']

#iris_df['label']은 Series형태이고, iris.target은 ndarray형태인데 상관없다!
for train_index, test_index in skfold.split(features, label):
    
    X_train,X_test=features[train_index], features[test_index]
    y_train,y_test=label[train_index], label[test_index] #label이 iris.target이면 ndarray, iris_df['label']이면 Series
    
    dt_clf.fit(X_train, y_train)
    pred=dt_clf.predict(X_test)
    
    n_iter+=1
    accuracy=np.round(accuracy_score(y_test, pred),4)
    train_size=X_train.shape[0]
    test_size=X_test.shape[0]
    
    print("\n#{0} 교차 검증 정확도 :{1},  학습 데이터 크기: {2}, 검증 데이터 크기: {3}".format(n_iter, accuracy, train_size, test_size))
    print("${0} 검증 세트 인덱스:{1}".format(n_iter, test_index))
    cv_accuracy.append(accuracy)
    
    print("\n## 교차 검증별 정확도:", np.round(cv_accuracy,4))
    print("## 평균 검증 정확도:", np.mean(cv_accuracy))
```

    
    #1 교차 검증 정확도 :0.98,  학습 데이터 크기: 100, 검증 데이터 크기: 50
    $1 검증 세트 인덱스:[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  50
      51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115]
    
    ## 교차 검증별 정확도: [0.98]
    ## 평균 검증 정확도: 0.98
    
    #2 교차 검증 정확도 :0.94,  학습 데이터 크기: 100, 검증 데이터 크기: 50
    $2 검증 세트 인덱스:[ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  67
      68  69  70  71  72  73  74  75  76  77  78  79  80  81  82 116 117 118
     119 120 121 122 123 124 125 126 127 128 129 130 131 132]
    
    ## 교차 검증별 정확도: [0.98 0.94]
    ## 평균 검증 정확도: 0.96
    
    #3 교차 검증 정확도 :0.98,  학습 데이터 크기: 100, 검증 데이터 크기: 50
    $3 검증 세트 인덱스:[ 34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  83  84
      85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 133 134 135
     136 137 138 139 140 141 142 143 144 145 146 147 148 149]
    
    ## 교차 검증별 정확도: [0.98 0.94 0.98]
    ## 평균 검증 정확도: 0.9666666666666667
    

### cross_val_score() : 교차 검증 API

cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')


```python
from sklearn.model_selection import cross_val_score, cross_validate

dt_clf=DecisionTreeClassifier(random_state=156)

iris=load_iris()
data=iris.data
label=iris.target

#교차검증폴드수=3
scores=cross_val_score(dt_clf,data, label, scoring='accuracy', cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
print('평균 검증 정확도:', np.round(np.mean(scores),4))
```

    교차 검증별 정확도: [0.98 0.94 0.98]
    평균 검증 정확도: 0.9667
    

**cross_validate()** 는 여러 개의 평가 지표를 반환하며, 학슫 데이터에 대한 성능 평가 지표와 수행시간도 제공

### GridSearchCV - 교차검증과 최적 하이퍼 파라미터 튜닝
grid_parameters={'max_depth':[1,2,3],'min_samples_split':[2,3]}<br/>
gridserachcv의 결과는 cv_results_에 저장

최고 성능 파라미터 -> **best_params_** <br/>
최고 성능 정확도   -> **best_score_**


```python
from sklearn.model_selection import GridSearchCV, train_test_split

iris=load_iris()
grid_parameters={'max_depth':[1,2,3],'min_samples_split':[2,3]}

X_train, X_test, y_train, y_test=train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
dtree=DecisionTreeClassifier()
grid_dtree=GridSearchCV(dtree, param_grid=grid_parameters, cv=3, refit=True)
grid_dtree.fit(X_train, y_train)

scores_df=pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'max_depth': 1, 'min_samples_split': 2}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'max_depth': 1, 'min_samples_split': 3}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'max_depth': 2, 'min_samples_split': 2}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'max_depth': 2, 'min_samples_split': 3}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'max_depth': 3, 'min_samples_split': 2}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{'max_depth': 3, 'min_samples_split': 3}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_dtree.best_score_))
```

    GridSearchCV 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
    GridSearchCV 최고 정확도:0.9750
    

refit=True이면, GridSearchCV가 최적 성능을 나타내는 하이퍼 파라미터로 Estimator를 학습해 **best_estimator_** 로 저장


```python
estimator=grid_dtree.best_estimator_
pred=estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
```

    테스트 데이터 세트 정확도: 0.9667
    
