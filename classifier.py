#!/usr/bin/python

import itertools
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.classification import f1_score, confusion_matrix


def get_alexa_rank_level(s):
    """
    Return the rank level based on alexa rank
    """
    if s is not None:
        if int(s) <= 5000:
            return 1
        if int(s) <= 10000:
            return 2
        if int(s) <= 50000:
            return 3
        if int(s) <= 100000:
            return 4
        if int(s) <= 500000:
            return 5
        if int(s) <= 1000000:
            return 6
        return 7
    else:
        return 8


def is_top_alexa(s):
    """
    Check if alexa rank is less or equal 1,000,000
    """
    if s is not None:
        if int(s) <= 1000000:
            return 1
    else:
        return 0


def model_tune_params(model, params, X, Y):
    """
    Tune a machine learning model an array of arrays of parameters
    """
    new_model = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        n_jobs=-1,
    )
    new_model.fit(X, Y)
    return new_model


# plot confusion matrix
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    """
    main function that train machine learning models to classify
    malicious URL
    """
    corpus = open('data/train.json')
    urldata = json.load(corpus, encoding="latin1")
    X_table = []
    X_header = [
        "host_length", "url_length", "domain_age", "domain_token_count",
        "path_length", "path_token_count", "is_top_alexa", "alexa_rank_level"
    ]
    Y = []
    Y_header = ["malicious_url"]

    logs = open("logs.txt", "w+")
    count_malicious_train = 0
    total_train = 0
    for row in urldata:
        data_row = []
        data_row.append(int(row["host_len"]))
        data_row.append(int(row["url_len"]))
        data_row.append(int(row["domain_age_days"]))
        data_row.append(int(row["num_domain_tokens"]))
        data_row.append(int(row["path_len"]))
        data_row.append(int(row["num_path_tokens"]))
        data_row.append(is_top_alexa(row["alexa_rank"]))
        data_row.append(get_alexa_rank_level(row["alexa_rank"]))

        X_table.append(data_row)
        Y.append(row["malicious_url"])
        total_train = total_train + 1
        if (row["malicious_url"] == 1):
            count_malicious_train = count_malicious_train + 1

    logs.write("--Training data--\n")
    logs.write("Total Samples: %d\n" % (total_train))
    logs.write("Malicious URLs: %d\n" % (count_malicious_train))
    logs.write(
        "Non-malicious URLs: %d\n" % (total_train - count_malicious_train))
    logs.write("\n")

    # create a panda dataframe
    X = pd.DataFrame(X_table, columns=X_header)
    Y = pd.DataFrame(Y, columns=Y_header)

    # split dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=100, stratify=Y, shuffle=True)

    logit = linear_model.LogisticRegression(random_state=100, dual=False)
    svm_clf = svm.SVC(random_state=100)
    rf = ensemble.RandomForestClassifier(random_state=100)

    logit_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'penalty': ('l2', 'l1')
    }

    svm_clf_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'gamma': [0.001, 0.0001, 0.00001],
        'kernel': ('poly', 'rbf', 'sigmoid')
    }
    rf_params = {
        'n_estimators': [10, 20, 30, 40, 50],
        'max_leaf_nodes': [50, 100, 150, 200],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy']
    }

    logit = model_tune_params(logit, logit_params, X_train, Y_train)
    svm_clf = model_tune_params(svm_clf, svm_clf_params, X_train, Y_train)
    rf = model_tune_params(rf, rf_params, X_train, Y_train)

    logit_pred = logit.predict(X_test)
    svm_clf_pred = svm_clf.predict(X_test)
    rf_pred = rf.predict(X_test)

    # f1 score on validation set
    logit_score = f1_score(Y_test, logit_pred)
    svm_clf_score = f1_score(Y_test, svm_clf_pred)
    rf_score = f1_score(Y_test, rf_pred)
    logs.write("--f1 score on validation set--\n")
    logs.write("Logistic Regression: {:5f}\n".format(logit_score))
    logs.write("Support Vector Machine: {:5f}\n".format(svm_clf_score))
    logs.write("Random Forest: {:5f}\n\n".format(rf_score))

    # confusion matrix on validation set
    classes = ["Not Malicious", "Malicious"]
    logit_matrix = confusion_matrix(Y_test, logit_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(
        logit_matrix, classes, normalize=False, title='logit')
    plt.savefig('images/logit.png')

    svm_clf_matrix = confusion_matrix(Y_test, svm_clf_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(
        svm_clf_matrix, classes, normalize=False, title='svm')
    plt.savefig('images/svm.png')

    rf_matrix = confusion_matrix(Y_test, rf_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(rf_matrix, classes, normalize=False, title='rf')
    plt.savefig('images/rf.png')

    # make prediction on classify.json
    classify_json = open('data/classify.json')
    classify_data = json.load(classify_json, encoding="latin1")

    # parse row data of classify_data
    classify_X = []
    classify_urls = []
    for row in classify_data:
        classify_urls.append(row["url"])
        data_row = []
        data_row.append(int(row["host_len"]))
        data_row.append(int(row["url_len"]))
        data_row.append(int(row["domain_age_days"]))
        data_row.append(int(row["num_domain_tokens"]))
        data_row.append(int(row["path_len"]))
        data_row.append(int(row["num_path_tokens"]))
        data_row.append(is_top_alexa(row["alexa_rank"]))
        data_row.append(get_alexa_rank_level(row["alexa_rank"]))
        classify_X.append(data_row)

    classify_X_df = pd.DataFrame(classify_X)
    classify_pred = svm_clf.predict(classify_X_df)

    # output the predict results to results.txt
    results = open("results.txt", "w+")
    for i in range(0, len(classify_pred)):
        # print "%s, %d" % (classify_urls[i], classify_pred[i])
        results.write("%s, %d\n" % (classify_urls[i], classify_pred[i]))
    results.close()

    count_malicious_classify = 0
    total_classify = 0
    for row in classify_pred:
        total_classify = total_classify + 1
        if row == 1:
            count_malicious_classify = count_malicious_classify + 1

    logs.write("--Classify data--\n")
    logs.write("Total Samples: %d\n" % (total_classify))
    logs.write("Malicious URL: %d\n" % (count_malicious_classify))
    logs.write("Non-malicious URL: %d\n" %
               (total_classify - count_malicious_classify))
    logs.write("\n")
    logs.close()

    # output the logs file to stdout
    with open("logs.txt", 'r') as logs:
        print logs.read()


if __name__ == "__main__":
    main()
