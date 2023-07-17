import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
class_names = data.target_names

# dataset = pd.read_csv("pulsar_stars.csv")
# cols = [col for col in list(dataset.columns) if 'target_class' not in col]
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# for feature in cols:
#     dataset[feature] = scaler.fit_transform(dataset[[feature]]) 
# X = dataset.drop(['target_class'], axis=1)
# y = dataset['target_class']
# class_names = ['not_pulsar', 'pulsar']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

from eval_classification import eval_classification
y_pred = clf_dt.predict(X_test)
y_pred_proba = clf_dt.predict_proba(X_test)

eval_classification( untrained_model=DecisionTreeClassifier(), n_splits=5,
                    class_names=class_names, 
                    X=X, y=y, 
                    y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, 
                    show=False, save=True, RESULTS_DIR=os.getcwd()+'/results')