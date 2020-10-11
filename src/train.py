import os
import argparse
import joblib

import pandas as pd
from sklearn import tree
from sklearn import metrics

from src import config, dispatcher
from src.metrics import f1


def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(columns=['kfold', 'y']).values
    y_train = df_train['y'].values

    x_valid = df_valid.drop(columns=['kfold', 'y']).values
    y_valid = df_valid['y'].values

    clf = dispatcher.MODELS[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    probs = clf.predict_proba(x_valid)[:, 1]

    # accuracy = metrics.accuracy_score(y_valid, preds)
    report = metrics.classification_report(y_valid, preds)
    auc = metrics.roc_auc_score(y_valid, probs)
    # print(f"Fold={fold}, Accuracy = {accuracy}")
    print(f"Fold={fold}, Report = {report}")
    print(f"Fold={fold}, AUC = {auc}")

    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT, f"dt_{model}_{fold}.bin")
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    run(fold=args.fold, model=args.model)
