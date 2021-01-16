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
 
