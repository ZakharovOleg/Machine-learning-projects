#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:17:38 2019

@author: OlegZakharov
"""

# Игнорирование предупреждений (Spyder (Python3.7))
import warnings
warnings.filterwarnings("ignore")

# Импортирование необходимых модулей и атрибутов
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot

# Загрузка набора данных
dataset = load_digits()
X = dataset.data
Y = dataset.target
# Разделение набора данных на тренировочные и тестовые части
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)
# Настройка параметров оценивания алгоритма
num_folds = 10
n_estimators = 100
scoring = 'accuracy'

###############################################################################
# Блиц-проверка алгоритмов машинного обучения (далее - алгоритмов) на исходных,
# необработанных, данных
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LSVC', LinearSVC()))
models.append(('SVC', SVC()))
models.append(('MLP', MLPClassifier()))
models.append(('BG', BaggingClassifier(n_estimators=n_estimators)))
models.append(('RF', RandomForestClassifier(n_estimators=n_estimators)))
models.append(('ET', ExtraTreesClassifier(n_estimators=n_estimators)))
models.append(('AB', AdaBoostClassifier(n_estimators=n_estimators,
                                        algorithm='SAMME')))
models.append(('GB', GradientBoostingClassifier(n_estimators=n_estimators)))
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
# Диаграмм размаха («ящик с усами»)
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
                                     ('LR', LogisticRegression())])))
pipelines.append(('SS_LDA', Pipeline([('Scaler', StandardScaler()),
                                      ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('SS_KNN', Pipeline([('Scaler', StandardScaler()),
                                      ('KNN', KNeighborsClassifier())])))
pipelines.append(('SS_CART', Pipeline([('Scaler', StandardScaler()),
                                       ('CART', DecisionTreeClassifier())])))
pipelines.append(('SS_NB', Pipeline([('Scaler', StandardScaler()),
                                     ('NB', GaussianNB())])))
pipelines.append(('SS_LSVC', Pipeline([('Scaler', StandardScaler()),
                                       ('LSVC', LinearSVC())])))
pipelines.append(('SS_SVC', Pipeline([('Scaler', StandardScaler()),
                                      ('SVC', SVC())])))
pipelines.append(('SS_MLP', Pipeline([('Scaler', StandardScaler()),
                                      ('MLP', MLPClassifier())])))
pipelines.append(('SS_BG', Pipeline
                  ([('Scaler', StandardScaler()),
                    ('BG', BaggingClassifier(n_estimators=n_estimators))])))
pipelines.append(('SS_RF', Pipeline
                  ([('Scaler', StandardScaler()),
                    ('RF', RandomForestClassifier(n_estimators=
                                                  n_estimators))])))
pipelines.append(('SS_ET', Pipeline
                  ([('Scaler', StandardScaler()),
                    ('ET', ExtraTreesClassifier(n_estimators=n_estimators))])))
pipelines.append(('SS_AB', Pipeline
                  ([('Scaler', StandardScaler()),
                    ('AB', AdaBoostClassifier(n_estimators=n_estimators,
                                              algorithm='SAMME'))])))
pipelines.append(('SS_GB', Pipeline
                  ([('Scaler', StandardScaler()),
                    ('GB',
                    GradientBoostingClassifier(n_estimators=n_estimators))])))
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
# Блиц-проверка алгоритмов на масштабированных исходных данных
# Масштабирование исходных данных в диапазон (0,1) (функция MinMaxScaler)
pipelines = []
pipelines.append(('MMS_LR', Pipeline([('Scaler', MinMaxScaler()),
                                      ('LR', LogisticRegression())])))
pipelines.append(('MMS_LDA', Pipeline([('Scaler', MinMaxScaler()),
                                       ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('MMS_KNN', Pipeline([('Scaler', MinMaxScaler()),
                                       ('KNN', KNeighborsClassifier())])))
pipelines.append(('MMS_CART', Pipeline([('Scaler', MinMaxScaler()),
                                        ('CART', DecisionTreeClassifier())])))
pipelines.append(('MMS_NB', Pipeline([('Scaler', MinMaxScaler()),
                                      ('NB', GaussianNB())])))
pipelines.append(('MMS_LSVC', Pipeline([('Scaler', MinMaxScaler()),
                                        ('LSVC', LinearSVC())])))
pipelines.append(('MMS_SVC', Pipeline([('Scaler', MinMaxScaler()),
                                       ('SVC', SVC())])))
pipelines.append(('MMS_MLP', Pipeline([('Scaler', MinMaxScaler()),
                                       ('MLP', MLPClassifier())])))
pipelines.append(('MMS_BG', Pipeline
                  ([('Scaler', MinMaxScaler()),
                    ('BG', BaggingClassifier(n_estimators=n_estimators))])))
pipelines.append(('MMS_RF', Pipeline
                  ([('Scaler', MinMaxScaler()),
                    ('RF', RandomForestClassifier(n_estimators=n_estimators))])))
pipelines.append(('MMS_ET', Pipeline
                  ([('Scaler', MinMaxScaler()),
                    ('ET', ExtraTreesClassifier(n_estimators=n_estimators))])))
pipelines.append(('MMS_AB', Pipeline
                  ([('Scaler', MinMaxScaler()),
                    ('AB', AdaBoostClassifier(n_estimators=n_estimators,
                                              algorithm='SAMME'))])))
pipelines.append(('MMS_GB', Pipeline
                  ([('Scaler', MinMaxScaler()),
                    ('GB', GradientBoostingClassifier(n_estimators=
                                                      n_estimators))])))
# Оценивание эффективности выполнения каждого алгоритма
scores_MMS = []
names_MMS = []
results_MMS = []
predictions_MMS = []
msg_MMS = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
    names_MMS.append(name)
    results_MMS.append(cv_results)
    m_fit = model.fit(X_train, Y_train)
    m_predict = model.predict(X_test)
    predictions_MMS.append(m_predict)
    m_score = model.score(X_test, Y_test)
    scores_MMS.append(m_score)
    msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(),
                           cv_results.std(), m_score)
    msg_MMS.append(msg)
    print(msg)
# ящик с усами (MinMaxScaler)
fig = pyplot.figure()
fig.suptitle('Сравнение результатов выполнения алгоритмов на масштаб. данных')
ax = fig.add_subplot(111)
red_square = dict(markerfacecolor='r', marker='s')
pyplot.boxplot(results_MMS, flierprops=red_square)
ax.set_xticklabels(names_MMS, rotation=45)
pyplot.show()

################################################################################
# Блиц-проверка алгоритмов на нормализованных исходных данных
# Нормализация исходных данных до единичной нормы (функция Normalizer)
pipelines = []
pipelines.append(('N_LR', Pipeline([('Scaler', Normalizer()),
                                    ('LR', LogisticRegression())])))
pipelines.append(('N_LDA', Pipeline([('Scaler', Normalizer()),
                                     ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('N_KNN', Pipeline([('Scaler', Normalizer()),
                                     ('KNN', KNeighborsClassifier())])))
pipelines.append(('N_CART', Pipeline([('Scaler', Normalizer()),
                                      ('CART', DecisionTreeClassifier())])))
pipelines.append(('N_NB', Pipeline([('Scaler', Normalizer()),
                                    ('NB', GaussianNB())])))
pipelines.append(('N_LSVC', Pipeline([('Scaler', Normalizer()),
                                      ('LSVC', LinearSVC())])))
pipelines.append(('N_SVC', Pipeline([('Scaler', Normalizer()),
                                     ('SVC', SVC())])))
pipelines.append(('N_MLP', Pipeline([('Scaler', Normalizer()),
                                     ('MLP', MLPClassifier())])))
pipelines.append(('N_BG', Pipeline
                  ([('Scaler', Normalizer()),
                    ('BG', BaggingClassifier
                     (n_estimators=n_estimators))])))
pipelines.append(('N_RF', Pipeline
                  ([('Scaler', Normalizer()),
                    ('RF', RandomForestClassifier
                     (n_estimators=n_estimators))])))
pipelines.append(('N_ET', Pipeline
                  ([('Scaler', Normalizer()),
                    ('ET', ExtraTreesClassifier
                     (n_estimators=n_estimators))])))
pipelines.append(('N_AB', Pipeline
                  ([('Scaler', Normalizer()),
                    ('AB', AdaBoostClassifier
                     (n_estimators=n_estimators, algorithm='SAMME'))])))
pipelines.append(('N_GB', Pipeline
                  ([('Scaler', Normalizer()),
                    ('GB', GradientBoostingClassifier
                     (n_estimators=n_estimators))])))
# Оценивание эффективности выполнения каждого алгоритма
scores_N = []
names_N = []
results_N = []
predictions_N = []
msg_N = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
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