 ### 프로젝트 기간(24.02.26 ~ 24.02.29)

# Kaggle contest Multi-Class prediction of obesity risk
### 프로젝트의 목적
- 주어진 피처 데이터들을 사용하여 비만도를 예측해보는 것
- Kaggle 대회라는 특성상 대회의 목적이 머신러닝 모델의 정확도를 최대한 높이는 것

### 본 프로젝트에서 사용한 환경
- 프로그래밍 언어 : python3(3.11)
- 사용 환경 : 구글 colab, kaggle notebook

### 주요 사용한 라이브러리
- NUmpy, Pandas, Scipy, Lgbmboost, Matplotlib
- sklearn
  - SelectFromModel(중요한 피처 칼럼을 랜덤포레스트를 이용해서 뽑아준다.)
  - XGBClassifier
  - LGBMClassifier
  - RandomForestClassifier
  - VotingClassifier
  - Pipeline
  - ColumnTransformer

 ### 데이터 전처리

    cat_cols_tr = list(train_data.select_dtypes(exclude = np.number).columns)
    num_cols_tr = list(train_data.select_dtypes(include = np.number).columns)

    cat_cols_ts = list(test_data.select_dtypes(exclude = np.number).columns)
    num_cols_ts = list(test_data.select_dtypes(include = np.number).columns)

위의 코드를 이용해서 숫자형 컬럼과 카테고리형 컬럼을 나눴다.

- 머신러닝 자체는 학습 데이터를 통해 규칙 즉 패턴을 찾아서 예측을 진행하는데 피처 컬럼중에 이러한 패턴이 존재하지 않는 'id'와 같은 피처 컬럼을 삭제해 주었다.

- pipeline을 이용했을 때
   - 카테고리형 데이터 : OneHotEncoder()
   - 연속형 수치 데이터 : StandardScaler()
- pipeline을 이용 하지 않았을 때
   - 카테고리형 데이터 : LabelEncoder() (머신러닝의 트리계열 분류 모델이기에 숫자에 대한 가중치나 중요도가 존재하지 않아서 OneHotEncoder를 사용하지 않음)
   - 연속형 수치 데이터 : StandardScaler()

### 데이터 전처리를 하면서 알게된 점 (이것이 LGBMClassifier의 장점)
- 내가 진행했던 트리계열 분류 머신러닝 모델들은 학습을 진행하기 전에 학습데이터와 학습이 끝난 후 예측을 하기 위한 테스트데이터 모두 수치형 데이터로 인코딩이 진행되어있어야 했다.
- 그렇기 때문에 데이터 전처리 과정에서 보면 train_data와 test_data를 모두 인코딩을 한 코드를 볼 수 있다.
- 하지만 LGBMClassifier는 특이하게 train_data만 인코딩을 진행해 준다면 학습을 한 이후에 테스트 데이터가 인코딩이 되어 있지 않더라도 자동으로 인코딩을 하고 예측을 해주었다.
- LGBMClassifier는 카테고리형 피처의 자동 변환과 최적의 분활을 해준다.(원-핫 인코딩 등을 사용하지 않고도 카테고리형 피처를 최적으로 변환하고 이에 따른 노드 분할을 수행한다. 그렇기에 인코딩을 해줄 필요가 없다.) 
  

### 수행 시도
##### 굉장히 여러 모델의 머신러닝 모델을 만들어서 시도했었다.
- xgboost 단일(하이퍼 파라미터 튜닝X) (Public Score:0.86452, Private Score:0.86623)
- xgboost 단일(하이퍼 파라미터 튜닝O) (Public Score:0.86199, Private Score:0.85738)
- RandomForest 단일(하이퍼 파라미터 튜닝 X) 
- RandomForest 단일(하이퍼 파라미터 튜닝 O) 
- xgboost, RandomForest를 soft voting(하이퍼 파라미터 튜닝 X) (Public Score:0.87933, Private Score:0.88312)
- xgboost, RandomForest를 soft voting(하이퍼 파라미터 튜닝 O) (Public Score:0.88403, Private Score:0.87897)
- lgbm 단일(하이퍼 파라미터 튜닝 O, pipeline사용) (Public Score:0.91112, Private Score:0.90164)
- 이외의 여러번의 시도 교차검증을 늘려보고, 줄요버고, 하이퍼 파라미터의 범위를 수도없이 조절해보고...(정확도가 계속 떨어져서 제출을 하지 않았다.)

### 결과

- 보통 대부분 Public score가 0.87정도에 머물러있었다. 즉 정확도가 87프로 정도 된다는 의미.
- RandomizedSearchCV를 통해서 넓은 범위의 하이퍼 파라미터를 조정해보고, 교차검증을 더 많이 진행하고, 불필요한 피처 데이터들을 라이브러리를 이용해서 그리고 시각화를 통해서 빼보고 머신러닝 모델을 학습 했지만 정확도는 올라가지 않았다.
- 결국에는 lgbm으로 하이퍼 파라미터를 조금 조정해주었을 때 가장 높은 점수가 나오게 되었다.(Public Score:0.91112, Private Score:0.90164)


 ## 이 대회를 통해 얻은 점
 - 나름 더 최고의 점수를 뽑기 위해서 굉장히 많은 시도를 했다. 하이퍼 파라미터를 계속 바꿔보고 여러개의 모델들을 소프트 보팅을 이용해서 학습을 진행하고 교차검증을 여러번 시도하고 최적의 피처컬럼만 뽑아서 해보고 했지만 정확도에서 유의미한 차이를 찾지 못했다.(계속 떨어졌다)
 - LGBM을 단일로 써서 하이퍼 파라미터를 튜닝해서 학습을 진행했을 때 가장 정확도가 잘나왔다.
 - 여기서 깨달은 점은 머신러닝을 진행하기 전에 더 많은 통계분석과 중요한 피처 컬럼을 뽑아내는 것 즉 데이터를 전처리 하는 과정이 머신러닝을 선택하고 학습하는 것보다 더 중요하다는 것을 알게 되었다.
 - 하이퍼 파라미터를 조정할 떄도 적정범위를 선택하는 것 또한 매우 중요하다는 것을 알았다.
 - 아직 머신러닝에 대해 잘 알지 못하고 파라미터들의 중요도와 어느정도 범위를 설정을 해야겠다는 감이 없기에 이러한 결과가 나온것 같다.
 - 다음에 또 이러한 대회를 한다면 신중하게 통계분석을 통해 피처 컬럼을 선택하고 왜 이러한 피처 컬럼을 선택해서 진행하게 되었는지에 대한 근거에 대해서 많이 생각해보아야 겠다는 생각을 하게 되었다.
