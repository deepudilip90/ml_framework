import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from dispatcher import MODELS
import joblib
import numpy as np

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')

def predict(path_test_data, model):
    df = pd.read_csv(path_test_data)
    test_idx = df['id'].values
    predictions = None

    for fold in range(5):
        df = pd.read_csv(path_test_data)
        label_encoders = joblib.load(os.path.join('models', f"{model}_{fold}_label_encoder.pkl"))
        cols = joblib.load(os.path.join('models', f"{model}_{fold}_columns.pkl"))
        for c in cols:
            lbl = label_encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join('models', f"{model}_{fold}.pkl"))

        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if fold == 0:
            predictions = preds
        else:
            predictions = predictions + preds
        
    predictions = predictions/5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['id', 'target'])
    return sub

        
if __name__ == '__main__':
    TEST_DATA = 'input/test.csv'
    MODEL='extratrees'

    submission = predict(TEST_DATA, MODEL)
    submission.to_csv(f"models/{MODEL}.csv", index=False )
