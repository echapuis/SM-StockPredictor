import os
from os.path import join
import time
import pandas as pd
import numpy as np
import pickle
import warnings
from matplotlib import pyplot as plt
import argparse
import logging
import csv
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from scipy.stats import describe

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters specify where/how files are saved. \n"
                                                 "This program takes as input a csv,tsv, or json file and outputs stuff.")
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help='saves intermediary files to save_dir')
    parser.add_argument('--load', '-l', action='store_true', default=True,
                        help='loads intermediary files automatically if available')
    parser.add_argument('--fresh', '-f', action='store_true', default=False,
                        help='sets loading and saving to false, overwriting existing files')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='prints program progress')
    args = parser.parse_args()
    if args.fresh:
        args.save = False
        args.load = False
    return args

def tune_Classifier(X_train, Y_train, iterations=30, n_estimators=10, verbose=False):

    if verbose: print("Tuning the Classifier...")

    # Set random seed to 0
    np.random.seed(0)

    # RA = ROC-AUC
    max_ra = -np.inf
    min_ra = np.inf
    max_param = 0
    result_pairs = []
    pct = 0
    start = time.time()
    for i in range(iterations):
        if verbose and i / 10. % 1 == 0:
            if pct != 0: print("Completed {}% ({}/{}) of iterations in {:.0f}:{:.0f} minutes".format(pct*10, i, iterations, (time.time()-start)//60, (time.time()-start)%60))
            pct += 1
        # "Pick a random value of C uniformly in the interval (1e-4, 1e4)"
        max_leaf_nodes = np.random.randint(2,100)
        min_samples_leaf = np.random.randint(2,100)
        min_samples_split = np.random.randint(2,100)
        # max_leaf_nodes = int(np.random.uniform(1, 1e5))

        # "Use 5-fold cross-validation to train the SVM"
        rfClassifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, min_samples_split=max_leaf_nodes)
        cvs = cross_val_score(rfClassifier, X_train, Y_train, scoring='roc_auc', cv=5)

        # "Estimate and record the ROC-AUC"
        iter_ra = np.mean(cvs)
        if iter_ra > max_ra:
            max_ra = iter_ra
            p1 = max_leaf_nodes
            p2 = min_samples_leaf
            p3 = min_samples_split
        if iter_ra < min_ra:
            min_ra = iter_ra

    # lift improvement
    lift = (max_ra - min_ra) / min_ra * 100

    return p1,p2,p3

def test_Classifier(X_train, Y_train, X_test, Y_test, max_leaf_nodes=100, min_samples_split=10, min_samples_leaf=1, random_state=0):
    # tests the classifier (yields predictions)

    # retrain SVM using all of the data
    classifier = RandomForestClassifier(n_estimators=1000, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    RFC_classifier = classifier.fit(X_train, Y_train)

    # generate predictions
    predY = RFC_classifier.predict_proba(X_test)[:,1]
    # print(Y_test)
    # print(predY)

    # record statistics
    fpr, tpr, thresholds = roc_curve(Y_test, predY)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# calculates the expected return period days in the future
def expected_return(data, period):
    data = data.copy()
    for i in range(0, len(data)):
        # print (i+min(period, len(data)-i), len(data))
        data.iloc[i,-1] = (data.iloc[i + min(period, len(data)-i-1),-1] - data.iloc[i,-1])
    return data

# Takes an entire df and normalizes it based on start
def normalize(data):
    data = data.copy()
    start = data.iloc[0,-1]
    for i in range(len(data)):
        data.iloc[i,-1] = data.iloc[i,-1]/start
    return data

# joins together feature df and label df
def merge_featurelabel(X,Y):
    D = X.join(Y, how='right')
    D = D.fillna(method='ffill')
    D = D.fillna(method='bfill')
    return D

# converts last column of dataframe to position labels based on threshold
def label_threshold(D, t, t2=''):
    D = D.copy()
    if t2 == '':
        for i in range(len(D)):
            if D.iloc[i,-1] > t:
                D.iloc[i,-1] = 1
            else:
                D.iloc[i, -1] = 0
    else:
        for i in range(len(D)):
            if D.iloc[i,-1] > t:
                D.iloc[i,-1] = 1
            elif D.iloc[i,-1] < t2:
                D.iloc[i,-1] = -1
            else:
                D.iloc[i, -1] = 0
    return D

# takes in pd dataframe with last column having position labels
def get_orders(Y, day_limit = 0):
    status = 0
    D = Y.copy()
    orders = []
    indices = list(D.index.values)[::-1]
    days = day_limit
    for i in range(len(D)):
        days -= 1
        # print(D.iloc[i,-1])
        # buy
        if D.iloc[i,-1] == 1 and status != 1 and days < 0:
            if status == 0:
                order = 1
            else:
                order = 2
            orders.append((indices[i], order))
            days = day_limit
            # print("BUY status/order: {},{}".format(status, order))
            status += order


        # sell
        elif D.iloc[i, -1] == -1 and status != -1 and days < 0:
            if status == 0:
                order = -1
            else:
                order = -2
            orders.append((indices[i], order))
            days = day_limit
            # print("SELL status/order: {},{}".format(status, order))
            status += order

        # hold
        elif D.iloc[i,-1] == 0 and status != 0 and days < 0:
            if status == 1:
                order = -1
            else:
                order = 1
            # print("hold: {},{}".format(status, order))
            orders.append((indices[i], order))
            days = day_limit
            # print("HOLD status/order: {},{}".format(status, order))
            status += order

    return orders #format is (order index, order - from -2 to 2)

def sim_portfolio(data, orders):
    orig = data
    D = data.copy()
    D.iloc[:,-1] = 1
    indices = sorted(list(D.index.values))
    # print(indices)
    N = len(indices)
    state = 0
    orders = orders.copy()
    order = orders.pop(0)
    D.iloc[0,-1] = 1
    for i in range(1,N):
        # print(state, D.iloc[i,-1])
        # print(state)
        index = indices.pop(0)
        if state == 1:
            D.iloc[i,-1] = orig.iloc[i,-1] - orig.iloc[i-1,-1] + D.iloc[i-1,-1]
        elif state == -1:
            D.iloc[i, -1] = -(orig.iloc[i, -1] - orig.iloc[i - 1, -1]) + D.iloc[i-1,-1]
        else:
            D.iloc[i,-1] = D.iloc[i-1, -1]
        if index >= order[0]:
            # print("order {}, oldstate {}".format(order[1],state))
            state += order[1]
            # print("newstate = ", state)
            if len(orders) != 0:
                order = orders.pop(0)
                # print(order)
            else:
                # print("final state: ", state)
                for i in range(i,N):
                    if state == 1:
                        D.iloc[i, -1] = orig.iloc[i, -1] - orig.iloc[i - 1, -1] + D.iloc[i - 1, -1]
                    elif state == -1:
                        D.iloc[i, -1] = -(orig.iloc[i, -1] - orig.iloc[i - 1, -1]) + D.iloc[i - 1, -1]
                    else:
                        D.iloc[i, -1] = D.iloc[i - 1, -1]
                break

    return D

def plot_ROC(fpr, tpr, roc_auc, title="ROC Curve"):
    # plots an ROC curve

    fig = plt.figure(figsize=(8,6))
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.rcParams["figure.figsize"] = [2,2]
    plt.show()

def ROC_experiment(X, Y,periods=[10], dataset = 'twitter', index='sp500'):

    X = merge_featurelabel(X,Y)

    buy_thresh = 0.0
    sell_thresh = 0.0

    X_train = X[X.index < 20120101]
    Y_train = X_train.ix[:, -1].as_matrix()
    X_train = X_train.ix[:, :-1].as_matrix()
    Y_train = np.divide(Y_train, Y_train[0])

    X_test = X[X.index >= 20120101]
    Y_test = X_test.ix[:, -1].as_matrix()
    X_test = X_test.ix[:, :-1].as_matrix()
    Y_test = np.divide(Y_test, Y_test[0])

    max_roc_auc = -np.inf
    max_fpr = -np.inf
    max_tpr = -np.inf
    max_period = periods[0]


    for period in periods:
        for i in range(0, len(Y_train) - period):
            Y_train[i] = (Y_train[i + period] - Y_train[i])
        Y_train[Y_train > buy_thresh] = 1
        Y_train[Y_train < sell_thresh] = 0

        for i in range(1, len(Y_test) - period):
            Y_test[i] = (Y_test[i + period] - Y_test[i])
        Y_test[Y_test > buy_thresh] = 1
        Y_test[Y_test < sell_thresh] = 0

        Y_train = np.asarray(Y_train, dtype='int')
        Y_test = np.asarray(Y_test, dtype='int')

        # print(Y_train)

        # p1, p2, p3 = tune_Classifier(X_train, Y_train, iterations=300)
        p1 = 52
        p2 = 93
        p3 = 52

        fpr, tpr, roc_auc = test_Classifier(X_train, Y_train, X_test, Y_test, max_leaf_nodes=p1, min_samples_leaf=p2,
                                            min_samples_split=p3)
        # print(roc_auc)

        if roc_auc > max_roc_auc:
            max_roc_auc = roc_auc
            max_fpr = fpr
            max_tpr = tpr
            max_period = period


    # print("max_params: {},{},{}".format(p1,p2,p3))
    print("ROC-AUC: {}".format(max_roc_auc))
    plot_ROC(max_fpr, max_tpr, max_roc_auc, title='{}-{}-{} ROC-AUC: {:.2f}'.format(dataset, index, max_period, max_roc_auc))

def plot_portfolio(port, bench, orders, dataset= 'twitter', index='sp500', period='', buy_thresh='', sell_thresh = '', day_limit=''):
    x1 = list(port.index.values)
    x1 = sorted([datetime.strptime(str(a), '%Y%m%d') for a in x1])
    y1 = list(port.iloc[:,-1].as_matrix())
    x2 = list(bench.index.values)
    x2 = sorted([datetime.strptime(str(a), '%Y%m%d') for a in x2])
    y2 = list(bench.iloc[:, -1].as_matrix())

    fig = plt.figure(figsize=(16,5))
    plt.subplot(121)
    # plt.axes([0,0.1,.5,.5])
    plt.plot(x1,y1,x2,y2)
    plt.legend(['Portfolio', 'Benchmark'])
    # plt.axis((x1[0], x1[-1], 0.7,1.3))
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title('Peformance of model using {} data and {} index'.format(dataset, index))

    curPos = 0
    # print(orders)
    for order in orders:
        day = datetime.strptime(str(order[0]), '%Y%m%d')
        ord = order[1]
        if ord > 0:
            color = 'green'
            curPos += ord
        elif ord < 0:
            color = 'red'
            curPos += ord
        if curPos == 0:
            color = 'black'
        plt.axvline(day, color=color, linewidth=1)

    fig.text(0.53,0.5, "Parameters:\nperiod = {}\nbuy threshold = {}\nsell threshold = {}\n# days between trades = {}".format(period, buy_thresh,sell_thresh, day_limit))
    plt.tight_layout(pad=2)
    plt.show()

def portfolio_experiment(X,Y, period=10, buy_thresh=0.01, sell_thresh=-0.01, day_limit=10, dataset= 'twitter', index='sp500'):

    D = merge_featurelabel(X,Y)
    D_train = D[D.index < 20120101].copy()
    # print(D_train.iloc[:10,-1])
    D_train = normalize(D_train)
    # print(D_train.iloc[:10, -1])
    D_train = expected_return(D_train, period)
    # print(D_train.iloc[:10, -1])
    D_train = label_threshold(D_train, buy_thresh, sell_thresh)
    # print(D_train.iloc[:10, -1])

    X_train = D_train.iloc[:,:-1].as_matrix()
    Y_train = D_train.iloc[:,-1].as_matrix()

    classifier = RandomForestClassifier(n_estimators=100, random_state=0, max_leaf_nodes=1000,
                                        min_samples_split=5, min_samples_leaf=1)

    # print(max(Y_train), min(Y_train))
    RFC_classifier = classifier.fit(X_train, Y_train)

    D_test = D[D.index >= 20120101].copy()
    D_test = normalize(D_test)
    D_test2 = D_test.copy()
    # print(D_test2.iloc[:10,-1])
    D_test2 = expected_return(D_test2, period)
    X_test = D_test2.iloc[:,:-1].as_matrix()

    Y_test = RFC_classifier.predict(X_test)

    print()

    D_test2.iloc[:,-1] = Y_test
    # print(max(D_test2.iloc[:,-1]), min(D_test2.iloc[:,-1]))

    orders = get_orders(D_test2, day_limit=day_limit)
    # print(len(orders))
    # print(D_test.iloc[:10,-1])
    # print(orders)
    D_test2 = sim_portfolio(D_test, orders)

    plot_portfolio(D_test2, D_test,orders, dataset= dataset, index=index, period=period, buy_thresh=buy_thresh, sell_thresh=sell_thresh, day_limit = day_limit)

if __name__ == '__main__':
    dataset = 'twitter'
    label_file = 'sp500.df'
    feature_folder = 'data/{}/finance-trained/y_n'.format(dataset)
    label_folder = 'data/indexes'
    feature_file = 'feature_dataframe.df'

    # feature_folder = 'data/demo/output'
    # feature_file = 'feature_dataframe.df'
    #
    # label_folder = 'data/indexes'
    # label_file = 'sp500.df'

    X = pickle.load(open(join(feature_folder, feature_file), 'rb'))
    Y = pickle.load(open(join(label_folder, label_file), 'rb'))

    period = 30
    buy_thresh = 0.0
    sell_thresh = -0.0
    day_limit = 1

    portfolio_experiment(X,Y,period, buy_thresh, sell_thresh, day_limit, dataset=dataset, index = label_file.split(".")[0])

    # ROC_experiment(X,Y, periods=[5], dataset='demo', index = label_file.split(".")[0])


#reddit - 52,93,52



