# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(data=wine.data, columns= wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)     
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

# GridSearch
param_grid = {
    "criterion" : ['gini', 'entropy'],
    "max_depth" : [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# HPO 및 Fitting
clf = DecisionTreeClassifier(random_state= 42)
grid_search = GridSearchCV(clf, param_grid, cv= 5)
grid_search.fit(X_train, y_train)

# HPO만들어진 모형의 정확도 계산 
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Grid :', accuracy)

# Feature Importance 시각화
importances = best_model.feature_importances_

plt.figure(figsize = (20,6))
plt.bar(range(len(importances)), importances, width=0.3)
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(range(len(importances)), X.columns, rotation = 45)
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

# XGboost 모델 생성
xgb_model = XGBClassifier(random_state=42)

# 하이퍼파라미터 범위지정
xgb_params = {
    "max_depth" : [3, 5, 7, 9, 15],
    "learning_rate" : [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}

# 하이퍼파라미터 최적화 및 학습
grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
best_xgb_model = grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print('Accuracy Grid :', accuracy_xgb)

# Feature Importance 시각화
importances_xgb = best_xgb_model.feature_importances_

plt.figure(figsize = (20,6))
plt.bar(range(len(importances_xgb)), importances_xgb, width=0.3)
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(range(len(importances_xgb)), X.columns, rotation = 45)
plt.show()