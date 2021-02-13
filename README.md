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
* Ensemble: 서로 다른/같은 머신러닝 알고리즘을 결합(Bagging, Boosting, Stacking)<br/>
Ensemble의 기본 알고리즘->DecisionTree(지니계수 이용해 데이터 세트 분할/과적합->하이퍼파라미터조정/Graphviz 이용해 시각화)
### Voting
* 서로 다른 알고리즘을 가진 여러 분류기가 투표를 통해 최종 예측 결과 결정
* 하드 보팅/소프트 보팅
### Bagging
* 모두 같은 유형의 알고리즘 기반/데이터 샘플링을 서로 다르게 가져가면서 학습 수행해 보팅 수행
* 랜덤 포레스트
### Boosting 
* GBM(Gradient Boosting Machine)
* XGBoost : 과적합규제, 조기중단, 병렬학습 -> 시간이 빨라짐
* LightGBM : 리프중심트리분할 -> 시간이 더 빨라짐, 메모리 사용량 적음 / (10000건 이하의 데이터 세트 사용시 과적합 가능성)
### Stacking
* 개별 알고리즘의 예측 결과 데이터 세트를 최종적인 메타 데이터 세트로 만들어 별도의 ML 알고리즘으로 최종 학습 수행
* 과적합 방지를 위해 CV 세트 기반의 스태킹

## 회귀
* 손실함수, 경사하강법: 수행시간의 문제 -> 확률적 경사 하강법(SGD)/미니 배치 확률적 경사 하강법
* LinearRegression -> coefficients(회귀계수) 
* 다항 회귀-PolynomialFeatures -> 곡선이 되지만 선형 회귀!!
* 규제 선형 모델-Ridge(L2규제), Lasso(L1규제), ElasticNet(L2+L1규제)
* 선형 회귀 모델을 위한 데이터 변환-타깃/피처의 스케일링/정규화(주로 로그변환)
* 로지스틱 회귀: 선형 회귀 방식을 분류에 적용한 알고리즘
* 회귀 트리: DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor - coef_속성이 없다!
* 스태킹 앙상블 모델을 통한 회귀 예측 가능
