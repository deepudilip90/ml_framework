import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from dispatcher import MODELS
import joblib


def train(path_train_data, path_test_data, fold_mapping, fold, model):
    print('Training for training fold: ', fold)
    df = pd.read_csv(path_train_data)
    df_test = pd.read_csv(path_test_data)
    
    train_df = df[df.kfold.isin(fold_mapping.get(fold))]
    valid_df = df[df.kfold==fold]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    print('Fitting Model: ', MODELS[model])
    clf = MODELS[model]
    
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f'models/{model}_{fold}_label_encoder.pkl')
    joblib.dump(clf, f'models/{model}_{fold}.pkl')
    joblib.dump(train_df.columns, f"models/{model}_{fold}_columns.pkl")

if __name__ == '__main__':
    TRAINING_DATA = 'input/train_folds.csv'
    TEST_DATA = 'input/test.csv'

    FOLD_MAPPING = {
        0: [1, 2, 3, 4],
        1: [0, 2, 3, 4],
        2: [0, 1, 3, 4],
        3: [0, 1, 2, 4],
        4: [0, 1, 2, 3]}
    FOLD = 0
    MODEL='extratrees'
    train(TRAINING_DATA, TEST_DATA, FOLD_MAPPING, 0, MODEL)
    train(TRAINING_DATA, TEST_DATA, FOLD_MAPPING, 1, MODEL)
    train(TRAINING_DATA, TEST_DATA, FOLD_MAPPING, 2, MODEL)
    train(TRAINING_DATA, TEST_DATA, FOLD_MAPPING, 3, MODEL)
    train(TRAINING_DATA, TEST_DATA, FOLD_MAPPING, 4, MODEL)



    


    

