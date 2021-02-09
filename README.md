# python-machinelearning-guide
## pandas
## Scikit-learn

### 1) 데이터 전처리
* fillna - 널 채워주기
* drop - 불필요 컬럼 제거
* 레이블 인코딩
  - LabelEncoder
  - OneHotEncoder(pandas의 get_dummies 이용 가능)
* 피처 스케일링(표준화, 정규화)
  - StandardScaler
  - MinMaxScaler

### 2) 학습/테스트 데이터 세트 분리
* train_test_split()
* 교차검증
  - K-fold
  - Stratified K-fold
  - cross_val_score()(교차검증 API)
  - GridSearchCV(교차검증 with 최적 하이퍼 파라미터 튜닝)
 
## 평가
* 정확도 -> accuracy_score()
* 오차행렬(TN, FP, FN, TP) -> confusion_matrix()
* 정밀도(precision) -> precision_score()
* 재현율(recall) -> recall_score()
* predict_proba() : 개별 데이터별로 예측 확률을 반환
* Binarizer(threshold)로 분류임계값에 기반하여 -> fit_transform(predict)
* precision_recall_curve() -> 반환값: thresholds, precision, recall

* F1스코어: 정밀도와 재현율 결합 -> f1_score()
* ROC 곡선: FPR이 변할 때, TPR의 변화 -> roc_curve() -> 반환값: thresholds, fpr, tpr
* AUC: ROC 곡선 밑의 면적(1에 가까울수록 좋은 수치) -> roc_auc_score()

KAGGLE: 전처리 -> 표준화 -> split -> 학습 -> 분류임계값 반영하여 예측 -> 평가

## 분류
* Ensemble: 서로 다른/같은 머신러닝 알고리즘을 결합<br/>
Ensemble의 기본 알고리즘->DecisionTree(지니계수 이용해 데이터 세트 분할/과적합->하이퍼파라미터조정)
### Bagging
### Boosting 
