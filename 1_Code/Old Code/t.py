"""
@author: Mohsen
GEM-ITH Ensemble - Fish data set
"""

# loading required packages
import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn import metrics
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import time
from sklearn.linear_model import LinearRegression
import warnings
import os

pd.set_option('display.max_columns', 500)
dataset = 'fish'

# loading the data set and preprocessing
fish = pd.read_csv('processed_train.csv', sep=',', header=None)
# fish.drop(columns=['Person_id'])
fish = fish.drop_duplicates()
fish = fish.reset_index(drop=True)
X = fish.drop(columns=[6])
Y = fish[6]
X_scaled = preprocessing.scale(X)
fish_scaled = X_scaled
fish_scaled = pd.DataFrame(fish_scaled, columns=X.columns)
fish_scaled[6] = Y

resampling = 5
objs = {}
test_results = {}
CV_results = {}
times = {}

kf = KFold(n_splits=5)

for iteration in range(resampling):
    print('Iteration ' + str(iteration + 1) + ' started...')

    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(fish_scaled.drop(columns=6),
                                                        fish_scaled[6], train_size=0.8)

    ## --------- Hyperparameters tuning (Bayesian Search) + ensemble ---------- ##

    objs[iteration] = pd.DataFrame(
        {'LASSO(a)': [], 'RF(n)': [], 'RF(MD)': [], 'XGB(g)': [], 'XGB(LR)': [], 'XGB(n)': [], 'XGB(MD)': [],
         'SVM(C)': [], 'SVM(g)': [], 'LASSO': [], 'RF': [], 'XGB': [], 'SVM-R': []})
    times[iteration] = pd.DataFrame({'LASSO': [], 'RF': [], 'XGB': [], 'SVM-R': [], 'BEM': [],
                                     'Stacked reg': [], 'Stacked RF': [], 'Stacked KNN': [], 'GEM-ITH': [], 'GEM': []})

    ## ---------
    start1 = time.time()

    # Bayesian search with 12 iterations to find candidate hyperparameter combinations
    max_evals = 12


    def objective_LASSO(params):
        L1_B = Lasso(**params)
        LASSO_B = L1_B.fit(x_train, y_train)
        LASSO_df_B = cross_val_predict(LASSO_B, x_train, y_train, cv=5)
        loss_LASSO = mse(y_train, LASSO_df_B)
        return {'loss': loss_LASSO, 'params': params, 'status': STATUS_OK}


    space_LASSO = {'alpha': hp.uniform('alpha', 10.0 ** -5.0, 1)}
    tpe_algorithm = tpe.suggest
    trials_LASSO = Trials()
    best_LASSO = fmin(fn=objective_LASSO, space=space_LASSO, algo=tpe.suggest,
                      max_evals=max_evals, trials=trials_LASSO)
    LASSO_param_B = pd.DataFrame({'alpha': []})
    for i in range(max_evals):
        LASSO_param_B.at[i, 'alpha'] = trials_LASSO.results[i]['params']['alpha']
    LASSO_param_B = pd.DataFrame(LASSO_param_B.alpha)


    def objective_RF(params):
        R1_B = RandomForestRegressor(**params)
        RF_B = R1_B.fit(x_train, y_train)
        RF_df_B = cross_val_predict(RF_B, x_train, y_train, cv=5)
        loss_RF = mse(y_train, RF_df_B)
        return {'loss': loss_RF, 'params': params, 'status': STATUS_OK}


    space_RF = {'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(4, 11, 1)])}
    tpe_algorithm = tpe.suggest
    trials_RF = Trials()
    best_RF = fmin(fn=objective_RF, space=space_RF, algo=tpe.suggest,
                   max_evals=max_evals, trials=trials_RF)
    RF_param_B = pd.DataFrame({'n_estimators': [], 'max_depth': []})
    for i in range(max_evals):
        RF_param_B.at[i, 'n_estimators'] = trials_RF.results[i]['params']['n_estimators']
        RF_param_B.at[i, 'max_depth'] = trials_RF.results[i]['params']['max_depth']
    RF_param_B = pd.DataFrame({'n_estimators': RF_param_B.n_estimators,
                               'max_depth': RF_param_B.max_depth})


    def objective_XGB(params):
        X1_B = XGBRegressor(objective='reg:linear', **params)
        XGB_B = X1_B.fit(x_train, y_train)
        XGB_df_B = cross_val_predict(XGB_B, x_train, y_train, cv=5)
        loss_XGB = mse(y_train, XGB_df_B)
        return {'loss': loss_XGB, 'params': params, 'status': STATUS_OK}


    space_XGB = {'gamma': hp.uniform('gamma', 5.0, 11.0),
                 'learning_rate': hp.uniform('learning_rate', 0.1, 0.6),
                 'n_estimators': hp.choice('n_estimators', np.arange(50, 151, 50)),
                 'max_depth': hp.choice('max_depth', np.arange(3, 10, 3))}
    tpe_algorithm = tpe.suggest
    trials_XGB = Trials()
    best_XGB = fmin(fn=objective_XGB, space=space_XGB, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials_XGB)
    XGB_param_B = pd.DataFrame({'gamma': [], 'learning_rate': [], 'n_estimators': [], 'max_depth': []})
    for i in range(max_evals):
        XGB_param_B.at[i, 'gamma'] = trials_XGB.results[i]['params']['gamma']
        XGB_param_B.at[i, 'learning_rate'] = trials_XGB.results[i]['params']['learning_rate']
        XGB_param_B.at[i, 'n_estimators'] = trials_XGB.results[i]['params']['n_estimators']
        XGB_param_B.at[i, 'max_depth'] = trials_XGB.results[i]['params']['max_depth']
    XGB_param_B = pd.DataFrame({'gamma': XGB_param_B.gamma,
                                'learning_rate': XGB_param_B.learning_rate,
                                'n_estimators': XGB_param_B.n_estimators,
                                'max_depth': XGB_param_B.max_depth})


    def objective_SVM(params):
        S1_B = svm.SVR(kernel='rbf', **params)
        SVM_B = S1_B.fit(x_train, y_train)
        SVM_df_B = cross_val_predict(SVM_B, x_train, y_train, cv=5)
        loss_SVM = mse(y_train, SVM_df_B)
        return {'loss': loss_SVM, 'params': params, 'status': STATUS_OK}


    space_SVM = {'C': hp.uniform('C', 0.01, 5.0),
                 'gamma': hp.uniform('gamma', 0.01, 0.55)}
    tpe_algorithm = tpe.suggest
    trials_SVM = Trials()
    best_SVM = fmin(fn=objective_SVM, space=space_SVM, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials_SVM)
    SVM_param_B = pd.DataFrame({'C': [], 'gamma': []})
    for i in range(max_evals):
        SVM_param_B.at[i, 'C'] = trials_SVM.results[i]['params']['C']
        SVM_param_B.at[i, 'gamma'] = trials_SVM.results[i]['params']['gamma']
    SVM_param_B = pd.DataFrame({'C': SVM_param_B.C,
                                'gamma': SVM_param_B.gamma})

    # defining optimization model
    def objective(x):
        return mse(y_train, (x[0] * LASSO_df +
                             x[1] * RF_df +
                             x[2] * XGB_df +
                             x[3] * SVM_df))

    def constraint1(x):
        return x[0] + x[1] + x[2] + x[3] - 1.0

    def constraint2(x):
        return LASSO_mse - objective(x)

    def constraint3(x):
        return RF_mse - objective(x)

    def constraint4(x):
        return XGB_mse - objective(x)

    def constraint5(x):
        return SVM_mse - objective(x)

    x0 = np.zeros(4)
    x0[0] = 1 / 4
    x0[1] = 1 / 4
    x0[2] = 1 / 4
    x0[3] = 1 / 4

    b = (0, 1.0)
    bnds = (b, b, b, b)
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    con4 = {'type': 'ineq', 'fun': constraint4}
    con5 = {'type': 'ineq', 'fun': constraint5}
    cons = [con1, con2, con3, con4, con5]

    kf = KFold(n_splits=5)

    # Initiating GEM-ITH algorithm to find optimal weights and hyperparameters
    for i in range(len(LASSO_param_B.alpha)):
        LASSO_df = pd.DataFrame()
        L1 = Lasso(alpha=LASSO_param_B.alpha[i])
        for train_index, test_index in kf.split(x_train, y_train):
            L1.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
            LASSO_df = pd.concat([LASSO_df, pd.DataFrame(L1.predict(np.array(x_train)[test_index]))])
        LASSO_mse = mse(y_train, LASSO_df)

        for j1 in range(len(RF_param_B)):
            RF_df = pd.DataFrame()
            R1 = RandomForestRegressor(n_estimators=RF_param_B.n_estimators[j1].astype(np.int64),
                                       max_depth=RF_param_B.max_depth[j1].astype(np.int64))
            for train_index, test_index in kf.split(x_train, y_train):
                R1.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
                RF_df = pd.concat([RF_df, pd.DataFrame(R1.predict(np.array(x_train)[test_index]))])
            RF_mse = mse(y_train, RF_df)

            for k1 in range(len(XGB_param_B)):
                XGB_df = pd.DataFrame()
                X1 = XGBRegressor(objective='reg:linear',
                                  gamma=XGB_param_B.gamma[k1],
                                  learning_rate=XGB_param_B.learning_rate[k1],
                                  n_estimators=XGB_param_B.n_estimators[k1].astype(np.int64),
                                  max_depth=XGB_param_B.max_depth[k1].astype(np.int64))
                for train_index, test_index in kf.split(x_train, y_train):
                    X1.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
                    XGB_df = pd.concat([XGB_df, pd.DataFrame(X1.predict(np.array(x_train)[test_index]))])
                XGB_mse = mse(y_train, XGB_df)

                for l1 in range(len(SVM_param_B)):
                    SVM_df = pd.DataFrame()
                    S1 = svm.SVR(kernel='rbf',
                                 C=SVM_param_B.C[l1],
                                 gamma=SVM_param_B.gamma[l1])
                    for train_index, test_index in kf.split(x_train, y_train):
                        S1.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
                        SVM_df = pd.concat([SVM_df, pd.DataFrame(S1.predict(np.array(x_train)[test_index]))])
                    SVM_mse = mse(y_train, SVM_df)

                    solution = minimize(objective, x0, method='SLSQP',
                                        options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds,
                                        constraints=cons)
                    x = solution.x
                    objs[iteration] = objs[iteration].append({'LASSO(a)': LASSO_param_B.alpha[i],
                                                              'RF(n)': RF_param_B.n_estimators[j1],
                                                              'RF(MD)': RF_param_B.max_depth[j1],
                                                              'XGB(g)': XGB_param_B.gamma[k1],
                                                              'XGB(LR)': XGB_param_B.learning_rate[k1],
                                                              'XGB(n)': XGB_param_B.n_estimators[k1].astype(np.int64),
                                                              'XGB(MD)': XGB_param_B.max_depth[k1].astype(np.int64),
                                                              'SVM(C)': SVM_param_B.C[l1],
                                                              'SVM(g)': SVM_param_B.gamma[l1],
                                                              'LASSO': LASSO_mse,
                                                              'RF': RF_mse,
                                                              'XGB': XGB_mse,
                                                              'SVM-R': SVM_mse,
                                                              'Weights': x,
                                                              'Initial Obj': objective(x0),
                                                              'Optimal Obj': objective(x)}, ignore_index=True)

    # making final base models predictions on the test set using the found hyperparameteers
    L_test = Lasso(alpha=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['LASSO(a)'])
    LASSO_test = L_test.fit(x_train, y_train)
    LASSO_preds_test = LASSO_test.predict(x_test)
    LASSO_mse_test = mse(y_test, LASSO_preds_test)

    R_test = RandomForestRegressor(
        max_depth=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['RF(MD)'],
        n_estimators=int(objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['RF(n)']))
    RF_test = R_test.fit(x_train, y_train)
    RF_preds_test = RF_test.predict(x_test)
    RF_mse_test = mse(y_test, RF_preds_test)

    X_test = XGBRegressor(objective='reg:linear',
                          gamma=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['XGB(g)'],
                          learning_rate=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :][
                              'XGB(LR)'],
                          max_depth=int(
                              objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['XGB(MD)']),
                          n_estimators=int(
                              objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['XGB(n)']))
    XGB_test = X_test.fit(x_train, y_train)
    XGB_preds_test = XGB_test.predict(x_test)
    XGB_mse_test = mse(y_test, XGB_preds_test)

    S_test = svm.SVR(kernel='rbf', C=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['SVM(C)'],
                     gamma=objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['SVM(g)'])
    SVM_test = S_test.fit(x_train, y_train)
    SVM_preds_test = SVM_test.predict(x_test)
    SVM_mse_test = mse(y_test, SVM_preds_test)

    # Aggregating base models predictions with the found optimal weights (GEM-ITH ensemble predictions)
    gemith_preds_test = objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['Weights'][
                            0] * LASSO_preds_test + \
                        objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['Weights'][
                            1] * RF_preds_test + \
                        objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['Weights'][
                            2] * XGB_preds_test + \
                        objs[iteration].loc[objs[iteration]['Optimal Obj'].idxmin(axis=1), :]['Weights'][
                            3] * SVM_preds_test
    gemith_mse_test = mse(y_test, gemith_preds_test)

    times[iteration]['GEM-ITH'] = [time.time() - start1]

    ## ------------------------------------ GEM - GRID SEARCH ------------------------------------- ##

    # performing Grid search to tune base models' hyperparameters
    start2 = time.time()

    grid_LASSO = {'alpha': np.linspace(10.0 ** -5.0, 1, 50)}
    L_gem_G = Lasso()
    search_LASSO_gem_G = GridSearchCV(L_gem_G, grid_LASSO, cv=5, scoring='neg_mean_squared_error')
    LASSO_df_gem_G = pd.DataFrame()
    for train_index, test_index in kf.split(x_train, y_train):
        search_LASSO_gem_G.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
        LASSO_df_gem_G = pd.concat(
            [LASSO_df_gem_G, pd.DataFrame(search_LASSO_gem_G.predict(np.array(x_train)[test_index]))])
    search_LASSO_gem_G.fit(x_train, y_train)
    LASSO_gem_G_mse = mse(y_train, LASSO_df_gem_G)
    LASSO_gem_G_test = search_LASSO_gem_G.predict(x_test)
    LASSO_gem_G_mse_test = mse(y_test, LASSO_gem_G_test)

    grid_RF = {'n_estimators': [100, 200, 500], 'max_depth': np.arange(4, 11, 1)}
    R_gem_G = RandomForestRegressor()
    search_RF_gem_G = GridSearchCV(R_gem_G, grid_RF, cv=5, scoring='neg_mean_squared_error')
    RF_df_gem_G = pd.DataFrame()
    for train_index, test_index in kf.split(x_train, y_train):
        search_RF_gem_G.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
        RF_df_gem_G = pd.concat([RF_df_gem_G, pd.DataFrame(search_RF_gem_G.predict(np.array(x_train)[test_index]))])
    search_RF_gem_G.fit(x_train, y_train)
    RF_gem_G_mse = mse(y_train, RF_df_gem_G)
    RF_gem_G_test = search_RF_gem_G.predict(x_test)
    RF_gem_G_mse_test = mse(y_test, RF_gem_G_test)

    grid_XGB = {'gamma': np.arange(5.0, 11.0), 'learning_rate': np.arange(0.1, 0.6),
                'n_estimators': np.arange(50, 151, 50), 'max_depth': np.arange(3, 10, 3)}
    X_gem_G = XGBRegressor(objective='reg:linear')
    search_XGB_gem_G = GridSearchCV(X_gem_G, grid_XGB, cv=5, scoring='neg_mean_squared_error')
    XGB_df_gem_G = pd.DataFrame()
    for train_index, test_index in kf.split(x_train, y_train):
        search_XGB_gem_G.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
        XGB_df_gem_G = pd.concat([XGB_df_gem_G, pd.DataFrame(search_XGB_gem_G.predict(np.array(x_train)[test_index]))])
    search_XGB_gem_G.fit(x_train, y_train)
    XGB_gem_G_mse = mse(y_train, XGB_df_gem_G)
    XGB_gem_G_test = search_XGB_gem_G.predict(x_test)
    XGB_gem_G_mse_test = mse(y_test, XGB_gem_G_test)

    grid_SVM = {'C': np.linspace(0.01, 5.0, 20), 'gamma': np.linspace(0.01, 0.55, 20)}
    S_gem_G = svm.SVR(kernel='rbf')
    search_SVM_gem_G = GridSearchCV(S_gem_G, grid_SVM, cv=5, scoring='neg_mean_squared_error')
    SVM_df_gem_G = pd.DataFrame()
    for train_index, test_index in kf.split(x_train, y_train):
        search_SVM_gem_G.fit(np.array(x_train)[train_index], np.array(y_train)[train_index])
        SVM_df_gem_G = pd.concat(
            [SVM_df_gem_G, pd.DataFrame(search_SVM_gem_G.predict(np.array(x_train)[test_index]))])
    search_SVM_gem_G.fit(x_train, y_train)
    SVM_gem_G_mse = mse(y_train, SVM_df_gem_G)
    SVM_gem_G_test = search_SVM_gem_G.predict(x_test)
    SVM_gem_G_mse_test = mse(y_test, SVM_gem_G_test)

    middle2 = time.time() - start2
    start2_gem = time.time()

    # defining optimization model
    def objective2(y):
        return mse(y_train, (y[0] * LASSO_df_gem_G +
                             y[1] * RF_df_gem_G +
                             y[2] * XGB_df_gem_G +
                             y[3] * SVM_df_gem_G))

    def constraint6(y):
        return y[0] + y[1] + y[2] + y[3] - 1.0

    def constraint7(y):
        return LASSO_gem_G_mse - objective2(y)

    def constraint8(y):
        return RF_gem_G_mse - objective2(y)

    def constraint9(y):
        return XGB_gem_G_mse - objective2(y)

    def constraint10(y):
        return SVM_gem_G_mse - objective2(y)


    y0 = np.zeros(4)
    y0[0] = 1 / 4
    y0[1] = 1 / 4
    y0[2] = 1 / 4
    y0[3] = 1 / 4
    b = (0, 1.0)
    bnds = (b, b, b, b)
    con6 = {'type': 'eq', 'fun': constraint6}
    con7 = {'type': 'ineq', 'fun': constraint7}
    con8 = {'type': 'ineq', 'fun': constraint8}
    con9 = {'type': 'ineq', 'fun': constraint9}
    con10 = {'type': 'ineq', 'fun': constraint10}
    cons2 = [con6, con7, con8, con9, con10]
    solution2 = minimize(objective2, y0, method='SLSQP',
                         options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds,
                         constraints=cons2)
    y = solution2.x

    # aggregating base models' predictions on the test set using the weights found via optimization (GEM ensemble predictions)
    cls_G_preds_test = y0[0] * LASSO_gem_G_test + y0[1] * RF_gem_G_test + y0[2] * XGB_gem_G_test + y0[
        3] * SVM_gem_G_test
    cls_G_mse_test = mse(y_test, cls_G_preds_test)

    times[iteration]['GEM'] = [time.time() - start2_gem + middle2]

    # Taking simple average from base models' predictions on the test set (BEM ensemble predictions)
    start2_cls = time.time()
    gem_G_preds_test = y[0] * LASSO_gem_G_test + y[1] * RF_gem_G_test + y[2] * XGB_gem_G_test + y[
        3] * SVM_gem_G_test
    gem_G_mse_test = mse(y_test, gem_G_preds_test)

    times[iteration]['BEM'] = [time.time() - start2_cls + middle2]

    ## -------------------------------- STACKING - GRID SEARCH -------------------------------- ##

    start3 = time.time()
    
    # making stacked ensemble models
    predsDF_G = pd.DataFrame()
    predsDF_G['LASSO'] = LASSO_df_gem_G[0].reset_index(drop=True)
    predsDF_G['RF'] = RF_df_gem_G[0].reset_index(drop=True)
    predsDF_G['XGB'] = XGB_df_gem_G[0].reset_index(drop=True)
    predsDF_G['SVM'] = SVM_df_gem_G[0].reset_index(drop=True)
    y_train.reset_index(inplace=True, drop=True)
    predsDF_G['y_train'] = y_train
    x_stacked_G = predsDF_G.drop(columns='y_train', axis=1)
    y_stacked_G = predsDF_G['y_train']
    testPreds_G = pd.DataFrame([LASSO_gem_G_test, RF_gem_G_test, XGB_gem_G_test, SVM_gem_G_test]).T
    testPreds_G.columns = ['LASSO', 'RF', 'XGB', 'SVM']

    middle3 = time.time() - start3

    start3_reg = time.time()
    stck_reg_G = LinearRegression()
    stck_reg_G.fit(x_stacked_G, y_stacked_G)
    stck_reg_G_preds_test = stck_reg_G.predict(testPreds_G)
    stck_reg_G_mse_test = mse(y_test, stck_reg_G_preds_test)
    times[iteration]['Stacked reg'] = [time.time() - start3_reg + middle2 + middle3]

    start3_rf = time.time()
    stck_rf_G = RandomForestRegressor()
    stck_rf_G.fit(x_stacked_G, y_stacked_G)
    stck_rf_G_preds_test = stck_rf_G.predict(testPreds_G)
    stck_rf_G_mse_test = mse(y_test, stck_rf_G_preds_test)
    times[iteration]['Stacked RF'] = [time.time() - start3_rf + middle2 + middle3]

    start3_knn = time.time()
    stck_knn_G = KNeighborsRegressor()
    stck_knn_G.fit(x_stacked_G, y_stacked_G)
    stck_knn_G_preds_test = stck_knn_G.predict(testPreds_G)
    stck_knn_G_mse_test = mse(y_test, stck_knn_G_preds_test)
    times[iteration]['Stacked KNN'] = [time.time() - start3_knn + middle2 + middle3]

    ## -------------------------------- GRID SEARCH FOR BASE LEARNERS -------------------------------- ##
    
    # performing grid search for base learners
    start4_lasso = time.time()
    grid_LASSO = {'alpha': np.linspace(10.0 ** -5.0, 1, 50)}
    L_G = Lasso()
    search_LASSO = GridSearchCV(L_G, grid_LASSO, cv=5, scoring='neg_mean_squared_error')
    search_LASSO.fit(x_train, y_train)
    LASSO_G_test = search_LASSO.predict(x_test)
    LASSO_G_mse_test = mse(y_test, LASSO_G_test)
    times[iteration]['LASSO'] = [time.time() - start4_lasso]

    start4_rf = time.time()
    grid_RF = {'n_estimators': [100, 200, 500], 'max_depth': np.arange(4, 11, 1)}
    R_G = RandomForestRegressor()
    search_RF = GridSearchCV(R_G, grid_RF, cv=5, scoring='neg_mean_squared_error')
    search_RF.fit(x_train, y_train)
    RF_G_test = search_RF.predict(x_test)
    RF_G_mse_test = mse(y_test, RF_G_test)
    times[iteration]['RF'] = [time.time() - start4_rf]

    start4_xgb = time.time()
    grid_XGB = {'gamma': np.arange(5.0, 11.0), 'learning_rate': np.arange(0.1, 0.6),
                'n_estimators': np.arange(50, 151, 50), 'max_depth': np.arange(3, 10, 3)}
    X_G = XGBRegressor(objective='reg:linear')
    search_XGB = GridSearchCV(X_G, grid_XGB, cv=5, scoring='neg_mean_squared_error')
    search_XGB.fit(x_train, y_train)
    XGB_G_test = search_XGB.predict(x_test)
    XGB_G_mse_test = mse(y_test, XGB_G_test)
    times[iteration]['XGB'] = [time.time() - start4_xgb]

    start4_svm = time.time()
    grid_SVM = {'C': np.linspace(0.01, 5.0, 20), 'gamma': np.linspace(0.01, 0.55, 20)}
    S_G = svm.SVR(kernel='rbf')
    search_SVM = GridSearchCV(S_G, grid_SVM, cv=5, scoring='neg_mean_squared_error')
    search_SVM.fit(x_train, y_train)
    SVM_G_test = search_SVM.predict(x_test)
    SVM_G_mse_test = mse(y_test, SVM_G_test)
    times[iteration]['SVM'] = [time.time() - start4_svm]

    ## ------------------------------ RESULTS ------------------------------ ##
    
    # recording test results
    test_results[iteration] = pd.DataFrame(data={'model': ['MSE'], 'LASSO': [LASSO_G_mse_test], 'RF': [RF_G_mse_test],
                                                 'XGB': [XGB_G_mse_test], 'SVM': [SVM_G_mse_test],
                                                 'GEM-ITH': [gemith_mse_test],
                                                 'GEM_G': [gem_G_mse_test], 'BEM_G': [cls_G_mse_test],
                                                 'stck_reg_G': [stck_reg_G_mse_test], 'stck_rf_G': [stck_rf_G_mse_test],
                                                 'stck_knn_G': [stck_knn_G_mse_test]})

    # averaging test results for 5 resamplings 
total_test_results = pd.concat([test_results[0].drop('model', axis=1), test_results[1].drop('model', axis=1),
                                test_results[2].drop('model', axis=1), test_results[3].drop('model', axis=1),
                                test_results[4].drop('model', axis=1)], axis=0)
test_results_mean = total_test_results.mean()
times_average = (times[0] + times[1] + times[2] + times[3] + times[4]) / resampling

GEM_params = pd.DataFrame(
    data={'model': ['parameter'], 'LASSO (alpha)': trials_LASSO.best_trial['result']['params']['alpha'],
          'RF (max_depth)': trials_RF.best_trial['result']['params']['max_depth'],
          'RF (n_estimators)': trials_RF.best_trial['result']['params']['n_estimators'],
          'XGB (gamma)': trials_XGB.best_trial['result']['params']['gamma'],
          'XGB (learning_rate)': trials_XGB.best_trial['result']['params']['learning_rate'],
          'XGB (n_estimators)': trials_XGB.best_trial['result']['params']['n_estimators'],
          'XGB (max_depth)': trials_XGB.best_trial['result']['params']['max_depth'],
          'SVM (C)': trials_SVM.best_trial['result']['params']['C'],
          'SVM (gamma)': trials_SVM.best_trial['result']['params']['gamma']})

grid_params = pd.DataFrame(data={'model': ['parameter'], 'LASSO (alpha)': search_LASSO.best_params_['alpha'],
                                 'RF (max_depth)': search_RF.best_params_['max_depth'],
                                 'RF (n_estimators)': search_RF.best_params_['n_estimators'],
                                 'XGB (gamma)': search_XGB.best_params_['gamma'],
                                 'XGB (learning_rate)': search_XGB.best_params_['learning_rate'],
                                 'XGB (n_estimators)': search_XGB.best_params_['n_estimators'],
                                 'XGB (max_depth)': search_XGB.best_params_['max_depth'],
                                 'SVM (C)': search_SVM.best_params_['C'],
                                 'SVM (gamma)': search_SVM.best_params_['gamma']})

objs[0].to_csv(str(dataset) + '_objs.csv')
GEM_params.to_csv(str(dataset) + '_gem_params.csv')
grid_params.to_csv(str(dataset) + '_grid_params.csv')
total_test_results.to_csv(str(dataset) + '_total_test_results.csv')
test_results_mean.to_csv(str(dataset) + '_test_results_mean.csv')
times_average.to_csv(str(dataset) + '_times_average.csv')

