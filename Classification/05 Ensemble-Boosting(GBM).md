# 앙상블 - GBM(boosting)
## GBM(Gradient Boosting Machine)
부스팅 알고리즘은 여러 개의 약한 학습기를 순차적으로 학습-예측하면서 잘못 예측한 데이터에 가중치 부여를 통해 오류를 개선해 나가면서 학습하는 방식-> 대표적으로 AdaBoost, Gradient Boost

수행시간이 오래걸린다!

### AdaBoosting
![image](https://user-images.githubusercontent.com/24853452/107628309-ce354a00-6ca3-11eb-8b56-c68ce73fddce.png)

가중치 업데이트를 경사 하강법 사용 -> **GBM**

경사하강법: 오류를 최소화하는 방향성을 가지고 반복적으로 가중치 값을 업데이트 하는 것


```python
from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df=pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df=feature_dup_df.reset_index()
    new_feature_name_df=pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name']=new_feature_name_df[['column_name','dup_cnt']].apply(lambda x:x[0]+'_'+str(x[1]) if x[1]>0 else x[0], axis=1)
    new_feature_name_df=new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

feature_name_df=pd.read_csv(r'C:\Users\user\Data_Handling\human_activity\features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])
feature_name=feature_name_df.iloc[:,1].values.tolist()
feature_dup_df=feature_name_df.groupby('column_name').count()

new_feature_name_df=get_new_feature_name_df(feature_name_df)
feature_name=new_feature_name_df.iloc[:,1].values.tolist()
X_train=pd.read_csv(r'C:\Users\user\Data_Handling\human_activity\train\X_train.txt', sep='\s+', names=feature_name)
X_test=pd.read_csv(r'C:\Users\user\Data_Handling\human_activity\test\X_test.txt', sep='\s+', names=feature_name)

y_train=pd.read_csv(r'C:\Users\user\Data_Handling\human_activity\train\y_train.txt', sep='\s+', header=None, names=['action'])
y_test=pd.read_csv(r'C:\Users\user\Data_Handling\human_activity\test\y_test.txt', sep='\s+', header=None, names=['action'])

start_time=time.time()
gb_clf=GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train,y_train)
gb_pred=gb_clf.predict(X_test)
gb_accuracy=accuracy_score(y_test, gb_pred)

print("GBM 정확도: {0:.4f}".format(gb_accuracy))
print("GBM 수행 시간: {0:.1f} 초".format(time.time()-start_time))
```

    GBM 정확도: 0.9389
    GBM 수행 시간: 652.5 초
    

GBM이 랜덤 포레스트보다는 예측 성능이 조금 뛰어난 경우가 많지만, 수행 시간이 오래 걸리고, 하이퍼 파라미터 튜닝 노력도 필요하다. <br/>

GBM은 멀티 CPU 코어 시스템을 사용하더라도 병렬 처리가 지원되지 않아서 대용량 데이터의 경우 학습에 매우 많은 시간이 필요하다.

### GridSearchCV로 하이퍼 파라미터 최적화
GBM의 하이퍼 파라미터
* loss : 경사하강법에서 사용할 비용함수
* learning_rate : GBM이 학습을 진행할 때마다 적용하는 학습률 
* n_estimators : weak learner의 개수, 개수가 많을수록 예측 성능이 일정 수준까지는 좋아질 수 있으나 개수가 많을수록 수행 시간이 오래걸린다.
* subsample : weak learner가 학습에 사용하는 데이터의 샘플링 비율, 기본값은 1이며 이는 전체 학습 데이터를 기반으로 학습한다는 의미


```python
from sklearn.model_selection import GridSearchCV

params={
    'n_estimators':[100,500],
    'learning_rate':[0.05,0.1]
}
grid_cv=GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)
print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

gb_pred=grid_cv.best_estimator_.predict(X_test)
gb_accuracy=accuracy_score(y_test, gb_pred)
print("GBM 정확도: {0:.4f}".format(gb_accuracy))
```

    Fitting 2 folds for each of 4 candidates, totalling 8 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed: 191.8min finished
    

    최적 하이퍼 파라미터:
     {'learning_rate': 0.1, 'n_estimators': 500}
    최고 예측 정확도: 0.9011
    GBM 정확도: 0.9420
    

총 3시간 걸림...
