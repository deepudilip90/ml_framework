import numpy as np
from collections import Counter

def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives.

    :param y_true: List of true values.
    :type y_true: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: Number of true positives in the list.
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if (yt == 1) & (yp == 1):
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: Number of true negatives in the list.
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    """
    Function to calculate True Negatives.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: Number of false positives in the list.
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    """
    Function to calculate True Negatives.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: Number of false negatives in the list.
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def precision(y_true, y_pred):
    """
    Function to calculate Precision.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: precision score.
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)

    if tp == fp == 0:
        return 0
    return tp/(tp + fp)


def recall(y_true, y_pred):
    """
    Function to calculate Recall.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    if tp == fn == 0:
        return 0

    return tp/(tp + fn)


def f1(y_true, y_pred):
    """
    Function to calculate Recall.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: f1 score
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p == r == 0:
        return 0

    f1 = 2 * p * r / (p + r)
    return f1


def tpr(y_true, y_pred):
    """
    Function to calculate True Positive Rate (tpr).

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: tpr (recall)
    """

    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    Function to calculate False Positive Rate (tpr).

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp)


def specificity(y_true, y_pred):
    """
    Function to calculate specificity / true negative rate (tnr).

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: specificity / tnr
    """
    return 1 - fpr(y_true, y_pred)


def log_loss(y_true, y_proba):
    """
    Function to calculate log-loss

    :param y_true: List of true values.
    :type y_true: list
    :param y_proba: List of predicted probabilites.
    :type y_proba: list
    :return: log loss from the predictions
    """

    epsilon = 1e-15
    loss = []

    for yt, yp in zip(y_true, y_proba):
        yp = np.clip(yp, epsilon, 1 - epsilon)
        cur_loss = -1 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        loss.append(cur_loss)

    return np.mean(loss)


def macro_precision(y_true, y_pred):
    """
    Function to calculate Macro Precision.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: macro precision score.
    """
    num_classes = len(np.unique(y_true))
    precision = 0
    for class_ in range(num_classes):
        cur_class_true = [1 if p == class_ else 0 for p in y_true]
        cur_class_pred = [1 if p == class_ else 0 for p in y_pred]
        tp = true_positive(cur_class_true, cur_class_pred)
        fp = false_positive(cur_class_true, cur_class_pred)
        class_precision = tp / (tp + fp)

        precision += class_precision

    precision /= num_classes

    return precision


def micro_precision(y_true, y_pred):
    """
    Function to calculate Micro Precision.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: macro precision score.
    """
    num_classes = len(np.unique(y_true))
    tp = 0
    fp = 0
    for class_ in range(num_classes):
        class_y_true = [1 if p == class_ else 0 for p in y_true]
        class_y_pred = [1 if p == class_ else 0 for p in y_pred]

        tp += true_positive(class_y_true, class_y_pred)
        fp += false_positive(class_y_true, class_y_pred)

    return tp / (tp + fp)


def weighted_precision(y_true, y_pred):
    """
    Function to calculate Macro Precision.

    :param y_true: List of true values.
    :type y_pred: list
    :param y_pred: List of predicted values.
    :type y_pred: list
    :return: macro precision score.
    """
    num_classes = len(np.unique(y_true))
    class_counts = Counter(y_true)

    precision = 0
    for class_ in range(num_classes):
        cur_class_true = [1 if p == class_ else 0 for p in y_true]
        cur_class_pred = [1 if p == class_ else 0 for p in y_pred]
        tp = true_positive(cur_class_true, cur_class_pred)
        fp = false_positive(cur_class_true, cur_class_pred)
        class_precision = tp / (tp + fp)

        precision += (class_counts[class_] * class_precision) / len(y_true)

    return precision


if __name__ == '__main__':
    l1 = [0,1,1,1,0,0,0,1]
    l2 = [0,1,0,1,0,1,0,0]

    # y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    # y_proba  = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

    print(weighted_precision(y_true, y_pred))
