#################################################################
#Copyright 2019 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
######################################################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit,StratifiedKFold,KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, Ridge
from sklearn.svm import LinearSVC, SVR, LinearSVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LassoLarsCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import time
import pdb
import time
import copy
from collections import Counter
#############################################################################
def accu(results, y_cv):
    return (results==y_cv).astype(int).sum(axis=0)/(y_cv.shape[0])
def rmse(results, y_cv):
    return np.sqrt(np.mean((results - y_cv)**2, axis=0))
################################################################################
def QuickML_Stacking(X_train, y_train, X_test='', modeltype='Regression',Boosting_Flag=False, 
                    scoring='', verbose=0):
    """
    Quickly build Stacks of multiple model results 
    Input must be a clean data set (only numeric variables, no categorical or string variables).
    """
    start_time = time.time()
    seed = 99
    if len(X_train) <= 100000 or X_train.shape[1] < 50:
        NUMS = 100
        FOLDS = 5
    else:
        NUMS = 200
        FOLDS = 10
    ## create Stacking models
    estimators = []
    ### This keeps tracks of the number of predict_proba columns generated by each model ####
    estimator_length = []
    if isinstance(X_test, str):
        no_fit = True
    else:
        no_fit = False
    if no_fit:
        #### This is where you don't fit the model but just do cross_val_predict ####
        if modeltype == 'Regression':
            if scoring == '':
                scoring = 'neg_mean_squared_error'
            scv = KFold(n_splits=FOLDS, random_state=seed, shuffle=True)
            if Boosting_Flag:
                ######    Bagging models if Bagging is chosen ####
                model4 = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                            n_estimators=NUMS,random_state=seed)
                results = cross_val_predict(model4,X_train,y_train, cv=scv,n_jobs=-1)
                estimators.append(('Bagging1',model4))
                estimator_length.append(1)
            elif Boosting_Flag is None:
                ####   Tree models if Linear chosen #####
                model5 = DecisionTreeRegressor(random_state=seed,min_samples_leaf=2)
                results = cross_val_predict(model5,X_train,y_train, cv=scv,n_jobs=-1)
                estimators.append(('Decision Trees',model5))
                estimator_length.append(1)
            else:
                ####   Linear Models if Boosting is chosen #####
                model6 = LassoCV(alphas=np.logspace(-10,-1,50), cv=scv,random_state=seed)
                results = cross_val_predict(model6,X_train,y_train, cv=scv,n_jobs=-1)
                estimators.append(('LassoCV Regularization',model6))
                estimator_length.append(1)
        else:
            n_classes = len(Counter(y_train))
            if scoring == '':
                scoring = 'accuracy'
            scv = StratifiedKFold(n_splits=FOLDS, random_state=seed, shuffle=True)
            if Boosting_Flag:
                ####   Linear Models if Boosting is chosen #####
                model4 = LinearDiscriminantAnalysis()
                results = cross_val_predict(model4,X_train,y_train, cv=scv,n_jobs=-1,
                                            method='predict_proba')
                estimators.append(('Linear Discriminant',model4))
                estimator_length.append(results.shape[1])
            elif Boosting_Flag is None:
                ####   Tree models if Linear chosen #####
                model6 = DecisionTreeClassifier(min_samples_leaf=2)
                results = cross_val_predict(model6,X_train,y_train, cv=scv,n_jobs=-1,
                                            method='predict_proba')
                estimators.append(('Decision Tree',model6))
                estimator_length.append(results.shape[1])
            else:
                ######    Naive Bayes models if Bagging is chosen ####
                if n_classes <= 2:
                    try:
                        model7 = GaussianNB()
                    except:
                        model7 = DecisionTreeClassifier(min_samples_leaf=2)
                else:
                    try:
                        model7 = MultinomialNB()
                    except:
                        model7 = DecisionTreeClassifier(min_samples_leaf=2)
                results = cross_val_predict(model7,X_train,y_train, cv=scv,n_jobs=-1,
                                            method='predict_proba')
                estimators.append(('Naive Bayes',model7))
                estimator_length.append(results.shape[1])
    else:
        #### This is where you fit the model and then predict ########
        if modeltype == 'Regression':
            if scoring == '':
                scoring = 'neg_mean_squared_error'
            scv = KFold(n_splits=FOLDS, random_state=seed, shuffle=True)
            if Boosting_Flag:
                ######    Bagging models if Bagging is chosen ####
                model4 = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                            n_estimators=NUMS,random_state=seed)
                results = model4.fit(X_train,y_train).predict(X_test)
                estimators.append(('Bagging1',model4))
                estimator_length.append(1)
            elif Boosting_Flag is None:
                ####   Tree models if Linear chosen #####
                model5 = DecisionTreeRegressor(random_state=seed,min_samples_leaf=2)
                results = model5.fit(X_train,y_train).predict(X_test)
                estimators.append(('Decision Trees',model5))
                estimator_length.append(1)
            else:
                ####   Linear Models if Boosting is chosen #####
                model6 = LassoCV(alphas=np.logspace(-10,-1,50), cv=scv,random_state=seed)
                results = model6.fit(X_train,y_train).predict(X_test)
                estimators.append(('LassoCV Regularization',model6))
                estimator_length.append(1)
        else:
            n_classes = len(Counter(y_train))
            if scoring == '':
                scoring = 'accuracy'
            scv = StratifiedKFold(n_splits=FOLDS, random_state=seed, shuffle=True)
            if Boosting_Flag:
                ####   Linear Models if Boosting is chosen #####
                model4 = LinearDiscriminantAnalysis()
                results = model4.fit(X_train,y_train).predict_proba(X_test)
                estimators.append(('Linear Discriminant',model4))
                estimator_length.append(results.shape[1])
            elif Boosting_Flag is None:
                ####   Tree models if Linear chosen #####
                model6 = DecisionTreeClassifier(min_samples_leaf=2)
                results = model6.fit(X_train,y_train).predict_proba(X_test)
                estimators.append(('Decision Tree',model6))
                estimator_length.append(results.shape[1])
            else:
                ######    Naive Bayes models if Bagging is chosen ####
                if n_classes <= 2:
                    try:
                        model7 = GaussianNB()
                    except:
                        model7 = DecisionTreeClassifier(min_samples_leaf=2)
                else:
                    try:
                        model7 = MultinomialNB()
                    except:
                        model7 = DecisionTreeClassifier(min_samples_leaf=2)
                results = model7.fit(X_train,y_train).predict_proba(X_test)
                estimators.append(('Naive Bayes',model7))
                estimator_length.append(results.shape[1])
    #stacks = np.c_[results1,results2,results3]
    estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
    estimator_names = [tuples[0] for tuples in estimators]
    #### Here is where we consolidate the estimator names and their results into one common list ###
    ls = []
    for x,y in dict(zip(estimator_names,estimator_length)).items():
        els = [x+str(eachy) for eachy in range(y)]
        ls += els
    if verbose == 1:
        print('    Time taken for Stacking: %0.1f seconds' %(time.time()-start_time))
    return ls, results
#########################################################
