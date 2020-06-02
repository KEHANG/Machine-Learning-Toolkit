#!/usr/bin/env python
# coding: utf-8

import optuna

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


## 1. Define Objective


def objective(trial):
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    
    classifier_name = trial.suggest_categorical("classifier", ["KNeighborsClassifier",
                                                               "SVC"])
    if classifier_name=="KNeighborsClassifier":
        n_neighbors = trial.suggest_int('n_neighbors', 3, 11)
        algorithm = trial.suggest_categorical("algorithm", 
                                              ["ball_tree",
                                                "kd_tree"])
        leaf_size = trial.suggest_int('leaf_size', 1, 50)
        metric = trial.suggest_categorical('metric', 
                                           ["euclidean","manhattan", 
                                            "chebyshev","minkowski"])
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               metric=metric,
                               )
    elif classifier_name=="SVC":
        C = trial.suggest_loguniform('C', 1e-10, 1)
        kernel = trial.suggest_categorical('kernel',['rbf','poly','rbf','sigmoid'])
        degree = trial.suggest_int('degree',1, 50)
        gamma = trial.suggest_loguniform('gamma',0.001,10000)
        clf = SVC(C=C, kernel=kernel, degree=degree,gamma=gamma)
    
    clf.fit(x_train,y_train)
    y_pred_test = clf.predict(x_test)
    loss = mean_squared_error(y_test,y_pred_test)
    print("Test Score:",clf.score(x_test,y_test))
    print("Train Score:",clf.score(x_train,y_train))
    print("\n=================")
    return loss


## 2. Local optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)


## 3. Distributed optimization
study = optuna.create_study(direction='minimize',
                            study_name='distributed-tuning',
							storage='mysql+pymysql://root:root@localhost:8888/ml_expts', 
                            load_if_exists=True)

study.optimize(objective, n_trials=10)