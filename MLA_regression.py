#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:15:12 2019

@author: OlegZakharov
"""

# Игнорирование предупреждений (Spyder (Python3.7))
import warnings
warnings.filterwarnings("ignore")

# Импортирование необходимых модулей и атрибутов
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot

# Загрузка набора данных
dataset = load_boston()
X = dataset.data
Y = dataset.target
# Разделение набора данных на тренировочные и тестовые части
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)
# Настройка параметров оценивания алгоритма
num_folds = 10
n_iter = 1000
n_estimators = 100
scoring = 'r2'

###############################################################################
# Блиц-проверка алгоритмов машинного обучения (далее - алгоритмов) на исходных,
# необработанных, данных
models = []
models.append(('LR', LinearRegression()))
models.append(('R', Ridge()))
models.append(('L', Lasso()))
models.append(('ELN', ElasticNet()))
models.append(('LARS', Lars()))
models.append(('BR', BayesianRidge(n_iter=n_iter)))
models.append(('KNR', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('LSVR', LinearSVR()))
models.append(('SVR', SVR()))
models.append(('ABR', AdaBoostRegressor(n_estimators=n_estimators)))
models.append(('BR', BaggingRegressor(n_estimators=n_estimators)))
models.append(('ETR', ExtraTreesRegressor(n_estimators=n_estimators)))
models.append(('GBR', GradientBoostingRegressor(n_estimators=n_estimators)))
models.append(('RFR', RandomForestRegressor(n_estimators=n_estimators)))

# Оценивание эффективности выполнения каждого алгоритма
scores = []
names = []
results = []
predictions = []
msg_row = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=
                                 scoring)
    names.append(name)
    results.append(cv_results)
    m_fit = model.fit(X_train, Y_train)
    m_predict = model.predict(X_test)
    predictions.append(m_predict)
    m_score = model.score(X_test, Y_test)
    scores.append(m_score)
    msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(),
                           cv_results.std(), m_score)
    msg_row.append(msg)
    print(msg)
# Диаграмма размаха («ящик с усами»)
fig = pyplot.figure()
fig.suptitle('Сравнение результатов выполнения алгоритмов')
ax = fig.add_subplot(111)
red_square = dict(markerfacecolor='r', marker='s')
pyplot.boxplot(results, flierprops=red_square)
ax.set_xticklabels(names, rotation=45)
pyplot.show()

###############################################################################
# Блиц-проверка алгоритмов на стандартизованных исходных данных
# Стандартизация исходных данных (функция StandardScaler)
pipelines = []
pipelines.append(('SS_LR', Pipeline([('Scaler', StandardScaler()),
                                    ('LR', LinearRegression())])))
pipelines.append(('SS_R', Pipeline([('Scaler', StandardScaler()),
                                     ('R', Ridge())])))
pipelines.append(('SS_L', Pipeline([('Scaler', StandardScaler()),
                                     ('L', Lasso())])))
pipelines.append(('SS_ELN', Pipeline([('Scaler', StandardScaler()),
                                      ('ELN', ElasticNet())])))
pipelines.append(('SS_LARS', Pipeline([('Scaler', StandardScaler()),
                                      ('LCV', Lars())])))
pipelines.append(('SS_BR', Pipeline([('Scaler', StandardScaler()),
                                      ('BR', BayesianRidge(n_iter=n_iter))])))
pipelines.append(('SS_KNR', Pipeline([('Scaler', StandardScaler()),
                                    ('KNR', KNeighborsRegressor())])))
pipelines.append(('SS_DTR', Pipeline([('Scaler', StandardScaler()),
                                      ('DTR', DecisionTreeRegressor())])))
pipelines.append(('SS_LSVR', Pipeline([('Scaler', StandardScaler()),
                                      ('LSVR', LinearSVR())])))
pipelines.append(('SS_SVR', Pipeline([('Scaler', StandardScaler()),
                                     ('SVR', SVR())])))
pipelines.append(('SS_ABR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('ABR', AdaBoostRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('SS_BR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('BR', BaggingRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('SS_ETR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('ETR', ExtraTreesRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('SS_GBR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('GBR', GradientBoostingRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('SS_RFR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('RFR', RandomForestRegressor
                             (n_estimators=n_estimators))])))
# Оценивание эффективности выполнения каждого алгоритма
scores_SS = []
names_SS = []
results_SS = []
predictions_SS = []
msg_SS = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
    names_SS.append(name)
    results_SS.append(cv_results)
    m_fit = model.fit(X_train, Y_train)
    m_predict = model.predict(X_test)
    predictions_SS.append(m_predict)
    m_score = model.score(X_test, Y_test)
    scores_SS.append(m_score)
    msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(),
                           cv_results.std(), m_score)
    msg_SS.append(msg)
    print(msg)
# ящик с усами (StandardScaler)
fig = pyplot.figure()
fig.suptitle('Сравнение результатов выполнения алгоритмов на стандарт. данных')
ax = fig.add_subplot(111)
red_square = dict(markerfacecolor='r', marker='s')
pyplot.boxplot(results_SS, flierprops=red_square)
ax.set_xticklabels(names_SS, rotation=45)
pyplot.show()

###############################################################################
# Блиц-проверка алгоритмов на нормализованных исходных данных
# Нормализация исходных данных до единичной нормы (функция Normalizer)
pipelines = []
pipelines.append(('N_LR', Pipeline([('Scaler', Normalizer()),
                                    ('LR', LinearRegression())])))
pipelines.append(('N_R', Pipeline([('Scaler', Normalizer()),
                                     ('R', Ridge())])))
pipelines.append(('N_L', Pipeline([('Scaler', Normalizer()),
                                     ('L', Lasso())])))
pipelines.append(('N_ELN', Pipeline([('Scaler', Normalizer()),
                                      ('ELN', ElasticNet())])))
pipelines.append(('N_LARS', Pipeline([('Scaler', Normalizer()),
                                      ('LCV', Lars())])))
pipelines.append(('N_BR', Pipeline([('Scaler', Normalizer()),
                                      ('BR', BayesianRidge(n_iter=n_iter))])))
pipelines.append(('N_KNR', Pipeline([('Scaler', Normalizer()),
                                    ('KNR', KNeighborsRegressor())])))
pipelines.append(('N_DTR', Pipeline([('Scaler', Normalizer()),
                                      ('DTR', DecisionTreeRegressor())])))
pipelines.append(('N_LSVR', Pipeline([('Scaler', Normalizer()),
                                      ('LSVR', LinearSVR())])))
pipelines.append(('N_SVR', Pipeline([('Scaler', Normalizer()),
                                     ('SVR', SVR())])))
pipelines.append(('N_ABR',
                  Pipeline([('Scaler', Normalizer()),
                            ('ABR', AdaBoostRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('N_BR',
                  Pipeline([('Scaler', Normalizer()),
                            ('BR', BaggingRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('N_ETR',
                  Pipeline([('Scaler', Normalizer()),
                            ('ETR', ExtraTreesRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('N_GBR',
                  Pipeline([('Scaler', Normalizer()),
                            ('GBR', GradientBoostingRegressor
                             (n_estimators=n_estimators))])))
pipelines.append(('N_RFR',
                  Pipeline([('Scaler', Normalizer()),
                            ('RFR', RandomForestRegressor
                             (n_estimators=n_estimators))])))
# Оценивание эффективности выполнения каждого алгоритма
scores_N = []
names_N = []
results_N = []
predictions_N = []
msg_N = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=
                                 scoring)
    names_N.append(name)
    results_N.append(cv_results)
    m_fit = model.fit(X_train, Y_train)
    m_predict = model.predict(X_test)
    predictions_N.append(m_predict)
    m_score = model.score(X_test, Y_test)
    scores_N.append(m_score)
    msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(),
                           cv_results.std(), m_score)
    msg_N.append(msg)
    print(msg)
# ящик с усами (Normalizer)
fig = pyplot.figure()
fig.suptitle('Сравнение результатов выполнения алгоритмов на норм. данных')
ax = fig.add_subplot(111)
red_square = dict(markerfacecolor='r', marker='s')
pyplot.boxplot(results_N, flierprops=red_square)
ax.set_xticklabels(names_N, rotation=45)
pyplot.show()