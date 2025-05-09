#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
#import seaborn as sns
#from sklearn.preprocessing import RobustScaler
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#import xgboost as xgb
#from sklearn.metrics import accuracy_score, f1_score
#import warnings
#warnings.filterwarnings("ignore")

# Load dataset
#df = pd.read_csv('../input/traffic-accident-prediction/dataset_traffic_accident_prediction1.csv')

# Handling missing values
#columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                        #'Driver_Age', 'Driver_Experience', 'Accident']

#for col in columns_with_missing:
    #df[col].fillna(df[col].median(), inplace=True)

#df['Weather'].fillna(df['Weather'].mode()[0], inplace=True)
#df['Road_Type'].fillna(df['Road_Type'].mode()[0], inplace=True)
#df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0], inplace=True)
#df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0], inplace=True)
#df['Road_Condition'].fillna(df['Road_Condition'].mode()[0], inplace=True)
#df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0], inplace=True)
#df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode()[0], inplace=True)

# Drop duplicates
#df = df.drop_duplicates()

# Feature engineering
#df['Age_vs_Experience'] = df['Driver_Age'] - df['Driver_Experience']
#df.drop(['Driver_Age', 'Driver_Experience'], axis=1, inplace=True)

# One-hot encoding for categorical variables
#df = pd.get_dummies(df, columns=['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 
                                 #'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'], drop_first=True)

# Define X and y
#X = df.drop('Accident', axis=1)
#y = df['Accident']

# Scale numeric features
#numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Age_vs_Experience']
#scaler = RobustScaler()
#X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Split dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
#xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

#param_xgb = {
   # 'n_estimators': [50, 100],
   # 'learning_rate': [0.1, 0.2],
   # 'max_depth': [3, 5],
   # 'subsample': [0.8, 1.0],
   # 'min_child_weight': [1, 3]
#}

#grid_search_xgb = GridSearchCV(xgb_classifier, param_xgb, cv=3)
#grid_search_xgb.fit(X_train, y_train)
#best_gs_xgb = grid_search_xgb.best_estimator_

#print("XGBoost:")
#print('  Train Score:', round(best_gs_xgb.score(X_train, y_train), 4))
#print('  Test Score :', round(best_gs_xgb.score(X_test, y_test), 4))

# Random Forest
#rf_classifier = RandomForestClassifier()
#param_rf = {
    #'max_depth': [3, 5],
    #'n_estimators': [100, 200],
    #'min_samples_split': [2, 5],
    #'min_samples_leaf': [1, 2]
#}

#grid_search_rf = GridSearchCV(rf_classifier, param_rf, cv=3)
#grid_search_rf.fit(X_train, y_train)
#best_gs_rf = grid_search_rf.best_estimator_

#print("\nRandom Forest:")
#print('  Train Score:', round(best_gs_rf.score(X_train, y_train), 4))
#print('  Test Score :', round(best_gs_rf.score(X_test, y_test), 4))

# KNN
#knn_classifier = KNeighborsClassifier()
#param_knn = {
    #'n_neighbors': range(3, 10),
    #'weights': ['uniform', 'distance'],
    #'algorithm': ['auto'],
    #'leaf_size': [20, 30]
#}

#grid_search_knn = GridSearchCV(knn_classifier, param_knn, cv=3)
#grid_search_knn.fit(X_train, y_train)
#best_gs_knn = grid_search_knn.best_estimator_

#print("\nK-Nearest Neighbors:")
#print('  Train Score:', round(best_gs_knn.score(X_train, y_train), 4))
#print('  Test Score :', round(best_gs_knn.score(X_test, y_test), 4))
