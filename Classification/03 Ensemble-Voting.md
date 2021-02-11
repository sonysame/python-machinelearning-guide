# 앙상블 - Voting

유형: Voting, Bagging, Boosting

* Bagging: Random Forest
* Boosting: 에이다 부스팅, Gradient Boosting, XGBoost, LightGBM
![image](https://user-images.githubusercontent.com/24853452/107627962-549d5c00-6ca3-11eb-82c7-c2838bf745e5.png)
Voting: 서로 다른 알고리즘을 가진 분류기를 결합<br/>
Bagging: 각각의 분류기가 모두 같은 유형의 알고리즘 기반이지만, 데이터 샘플링을 서로 다르게 가져가면서 학습 수행, 개별 Classifer에 할당된 학습 데이터는 원본 학습 데이터를 샘플링해 추출한다(Bootstrapping), 배깅 방식은 중첩을 허용한다.<br/>
Boosting: 여러 개의 분류기가 순차적으로 학습을 수행하되, 다음 분류기에게는 가중치를 부여하면서 학습과 예측을 진행<br/>
Stacking: 여러가지 다른 모델의 예측 결괏값을 다시 학습 데이터로 만들어서 다른 모델로 재학습시켜 결과를 예측하는 방법


## 보팅(Hard Voting, Soft Voting)
하드보팅: 예측한 결괏값들중 다수의 분류기가 결정한 예측값을 최종 보팅 겨로가값으로 선정<br/>
소프트보팅: 분류기들의 레이블 값 결정 확률을 모두 더하고 이를 평균해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값으로 선정 -> 일반적으로 소프트 보팅 사용!
![image](https://user-images.githubusercontent.com/24853452/107627979-59faa680-6ca3-11eb-80e8-04b9d8d94da6.png)

### VotingClassifier



```python
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer=load_breast_cancer()
data_df=pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head(3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>




```python
lr_clf=LogisticRegression(max_iter=500)
knn_clf=KNeighborsClassifier(n_neighbors=8)

vo_clf=VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)], voting='soft')
X_train, X_test, y_train, y_test=train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

vo_clf.fit(X_train, y_train)
pred=vo_clf.predict(X_test)
print("Voting 분류기 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

classifiers=[lr_clf, knn_clf]

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred=classifier.predict(X_test)
    class_name=classifier.__class__.__name__
    print("{0} 정확도:{1:.4f}".format(class_name, accuracy_score(y_test, pred)))
```

    Voting 분류기 정확도:0.9561
    LogisticRegression 정확도:0.9474
    KNeighborsClassifier 정확도:0.9386
    

보통, 단일 기반 분류기보다 이를 결합한 앙상블이 정확도가 조금 더 높게 나온다. 아닐 수도 있다!
