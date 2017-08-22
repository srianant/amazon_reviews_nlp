'''
    File name         : sentiment_analysis.py
    File Description  : NLP sentiment analysis routines
    Author            : Srini Ananthakrishnan
    Date created      : 12/14/2016
    Date last modified: 12/14/2016
    Python Version    : 2.7
'''

# Standard import
from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.extmath import density
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def plot_roc_curve(clf, X_train, X_test, y_train, y_test, y_pred, title, prob):
    """Function plots ROC curve
    Args:
        clf     : classifier model
        X_train : X training text data
        y_train : y training class/label data
        X_test  : X test text data
        y_test  : y test class/label data
        prob    : boolen to identify probability calculation
    Return:
        The return value. None
    """
    # Predict probablities
    if prob:
        y_score = clf.predict_proba(X_test)
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    else:
        y_score = clf.fit(X_train, y_train).decision_function(X_test)
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_score)

    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def binarize_overall(rating):
    """Function binarize ratings
    Args:
        rating     : rating from 1 to 5
    Return:
        The return value. 1 if > 3 else 0
    """
    if rating > 3:
        return 1
    elif rating <= 3:
        return 0
        
def tfidf_vectorize(X_train, X_test):
    """Function TF-IDF vectorizer
    Args:
        X_train : X training text data
        X_test  : X test text data
    Return:
        The return value. Vectorized X_train and X_test
    """
    # TF-IDF vectorize bigram range
    vectorizer = TfidfVectorizer(sublinear_tf=True, 
                                 max_df=0.5,
                                 ngram_range=(1, 2),
                                 stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    t0 = time()
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, X_train.shape[0] / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, X_test.shape[0] / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    print(X_train.shape,X_test.shape)
    return X_train, X_test
    
def classify(clf,X_train,X_test,y_train,y_test, title, prob):
    """Function classify fit/predict and evaluate metrics like accuracy
    Args:
        clf     : classifier model
        X_train : X training text data
        y_train : y training class/label data
        X_test  : X test text data
        y_test  : y test class/label data
        prob    : boolen to identify probability calculation
    Return:
        The return value. clf_descr, score, train_time, test_time
    """
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    target_names = ['NEG','POS']
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("clf.coef_:\n",clf.coef_)

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                        target_names=['POS','NEG']))

    #if opts.print_cm:
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    #clf_descr = str(clf).split('(')[0]
    clf_descr = title
    
    # Plot ROC curve
    plot_roc_curve(clf, X_train, X_test, y_train, y_test, pred, title, prob)
    return clf_descr, score, train_time, test_time

def kFold_cross_val(df, pipeline):
    """Function kFold cross validate and print score/confusion matrix
    Args:
        df       : input pandas data frame
        pipeline : input sklearn pipeline model
    Return:
        The return value. None
    """
    k_fold = KFold(n=len(df), n_folds=6)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = df.iloc[train_indices]['reviewText'].values
        train_y = df.iloc[train_indices]['sentiment'].values

        test_text = df.iloc[test_indices]['reviewText'].values
        test_y = df.iloc[test_indices]['sentiment'].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=1)
        scores.append(score)

    print('Total reviews classified:', len(df))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)

def plot_results(results):
    """Function plots summary results
    Args:
        results  : results contains score, test-time
    Return:
        The return value. None
    """
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()
