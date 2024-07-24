from catboost import CatBoostClassifier
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin
from tabddpm.sample import sample
import tabddpm.lib as lib
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, classification_report, roc_curve, confusion_matrix, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_metrics_updated(y_true, y_pred, y_pred_prob, metrics=[]):
    out = []
    for met in metrics:
      if met == roc_auc_score or met == roc_curve:
        out.append(met(y_true, y_pred_prob))
      else:
        out.append(met(y_true, y_pred))
    return out

def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)

def train_test_catboost():
    catboost_config = load_json(f'tuned_models/catboost/churn2_cv.json')
    model = CatBoostClassifier(
            loss_function="Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=10,
            class_names=["0", "1"]
        )
    
    real_data_directory = 'data/churn2'
    train_X_cat_array = np.load(f'{real_data_directory}/X_cat_train.npy')
    train_X_cat = pd.read_csv(train_X_cat_array, columns = ['cat_1, cat_2, cat_3, cat_4'])
    train_X_cat_array = np.load(f'{real_data_directory}/X_num_train.npy')
    train_X_num = pd.read_csv(train_X_cat_array, columns = ['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7'])
    train_y_array = np.load(f'{real_data_directory}/y_train.npy')
    train_y = pd.read_csv(train_y_array, columns = ['y'])

    val_X_cat_array = np.load(f'{real_data_directory}/X_cat_val.npy')
    val_X_cat = pd.read_csv(val_X_cat_array, columns = ['cat_1, cat_2, cat_3, cat_4'])
    val_X_cat_array = np.load(f'{real_data_directory}/X_num_val.npy')
    val_X_num = pd.read_csv(val_X_cat_array, columns = ['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7'])
    val_y_array = np.load(f'{real_data_directory}/y_val.npy')
    val_y = pd.read_csv(val_y_array, columns = ['y'])

    test_X_cat_array = np.load(f'{real_data_directory}/X_cat_test.npy')
    test_X_cat = pd.read_csv(test_X_cat_array, columns = ['cat_1, cat_2, cat_3, cat_4'])
    test_X_cat_array = np.load(f'{real_data_directory}/X_num_test.npy')
    test_X_num = pd.read_csv(test_X_cat_array, columns = ['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7'])
    test_y_array = np.load(f'{real_data_directory}/y_test.npy')
    test_y = pd.read_csv(test_y_array, columns = ['y'])


    raw_config = lib.load_config('exp/churn2/ddpm_cb_best/config.toml')
    X_num, X_cat, y_gen = sample(
                        num_samples=raw_config['sample']['num_samples'],
                        batch_size=raw_config['sample']['batch_size'],
                        disbalance=raw_config['sample'].get('disbalance', None),
                        **raw_config['diffusion_params'],
                        parent_dir=raw_config['parent_dir'],
                        real_data_path=raw_config['real_data_path'],
                        model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
                        model_type=raw_config['model_type'],
                        model_params=raw_config['model_params'],
                        T_dict=raw_config['train']['T'],
                        num_numerical_features=raw_config['num_numerical_features'],
                        device=DEVICE,
                        seed=raw_config['sample'].get('seed', 0),
                        change_val=False
                    )
    X_cat_df = pd.DataFrame(X_cat, columns=['cat_1, cat_2, cat_3, cat_4'])
    X_num_df = pd.DataFrame(X_num, columns=['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7'])
    y_gen_df = pd.DataFrame(y_gen, columns=['y'])
    generated_df = pd.concat([X_num_df, X_cat_df, y_gen_df], axis=1)
    raw_train_df = pd.concat([train_X_num, train_X_cat, train_y], axis=1)
    raw_test_df = pd.concat([test_X_num, test_X_cat, test_y], axis=1)
    raw_val_df = pd.concat([val_X_num, val_X_cat, val_y], axis=1)

    train_df = pd.concat([raw_train_df, generated_df], axis=0)

    train_X = train_df.drop(columns=['y'])
    train_Y = train_df['y']
    eval_X = raw_val_df.drop(columns=['y'])
    eval_Y = raw_val_df['y']
    test_X = raw_test_df.drop(columns=['y'])
    test_Y = raw_test_df['y']

    model.fit(
        train_X, train_Y,
        eval_set=(eval_X, eval_Y),
        verbose=100
    )
    accuracy = model.score(test_X, test_Y)
    predictions = model.predict(test_Y)
    predicted_probabilities = model.predict_proba(test_Y)
    acc, roc_auc, f1, prec, rec, roc_curve_result = calc_metrics_updated(test_Y, predictions, predicted_probabilities, metrics=[balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve])
    print(f'Accuracy: ',accuracy)
    print(f'OOF ACC: {np.mean(acc)}')
    print(f'OOF ROC AUC Score: {np.mean(roc_auc)}')
    print(f'OOF F1 Score: {np.mean(f1):.4f}')
    print(f'OOF Prec Score: {np.mean(prec):.4f}')
    print(f'OOF Recall Score: {np.mean(rec):.4f}')

if __name__ == "__main__":
    train_test_catboost()