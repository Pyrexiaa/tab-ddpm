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
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_pred_prob = np.array(y_pred_prob)

    if y_pred_prob.ndim > 1:
        y_pred_prob = y_pred_prob[:, 1]

    print(len(y_true))
    print(len(y_pred))
    print(len(y_pred_prob))

    assert len(y_true) == len(y_pred), f"Lengths of y_true {len(y_true)} and y_pred {len(y_pred)} do not match."
    assert len(y_true) == len(y_pred_prob), f"Lengths of y_true {len(y_true)} and y_pred_prob {len(y_pred_prob)} do not match."

    out = []
    for met in metrics:
      if met == roc_auc_score or met == roc_curve:
        out.append(met(y_true, y_pred_prob))
      else:
        out.append(met(y_true, y_pred))
    return out

def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)

def train_test_catboost(save_dir, dataset = 'churn2', categorical=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    catboost_config = load_json(f'tuned_models/catboost/{dataset}_cv.json')
    model = CatBoostClassifier(
            loss_function="Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=10,
            class_names=["0", "1"]
        )
    
    y_column_names = ['y']
    real_data_directory = f'data/{dataset}'
    if categorical:
        train_X_cat_array = np.load(f'{real_data_directory}/X_cat_train.npy')
        cat_columns = train_X_cat_array.shape[1]
        cat_column_names = [f'cat_{i+1}' for i in range(cat_columns)]
        train_X_cat = pd.DataFrame(train_X_cat_array, columns=cat_column_names)
        # train_X_cat.columns = cat_column_names
        
    train_X_num_array = np.load(f'{real_data_directory}/X_num_train.npy')
    num_columns = train_X_num_array.shape[1]
    num_column_names = [f'num_{i+1}' for i in range(num_columns)]
    train_X_num = pd.DataFrame(train_X_num_array, columns=num_column_names)
    # train_X_num.columns = num_column_names
    train_y_array = np.load(f'{real_data_directory}/y_train.npy')
    train_y = pd.DataFrame(train_y_array, columns=y_column_names)
    # train_y.columns = y_column_names

    if categorical:
        val_X_cat_array = np.load(f'{real_data_directory}/X_cat_val.npy')
        val_X_cat = pd.DataFrame(val_X_cat_array, columns=cat_column_names)
        # val_X_cat.columns = cat_column_names
    val_X_num_array = np.load(f'{real_data_directory}/X_num_val.npy')
    val_X_num = pd.DataFrame(val_X_num_array, columns=num_column_names)
    # val_X_num.columns = num_column_names
    val_y_array = np.load(f'{real_data_directory}/y_val.npy')
    val_y = pd.DataFrame(val_y_array, columns=y_column_names)
    # val_y.columns = y_column_names

    if categorical:
        test_X_cat_array = np.load(f'{real_data_directory}/X_cat_test.npy')
        test_X_cat = pd.DataFrame(test_X_cat_array, columns=cat_column_names)
        # test_X_cat.columns = cat_column_names
    test_X_num_array = np.load(f'{real_data_directory}/X_num_test.npy')
    test_X_num = pd.DataFrame(test_X_num_array, columns=num_column_names)
    # test_X_num.columns = num_column_names
    test_y_array = np.load(f'{real_data_directory}/y_test.npy')
    test_y = pd.DataFrame(test_y_array, columns=y_column_names)
    # test_y.columns = y_column_names


    raw_config = lib.load_config(f'exp/{dataset}/ddpm_cb_best/config.toml')
    if categorical:
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
    else:
        X_num, y_gen = sample(
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
    if categorical:
        X_cat_df = pd.DataFrame(X_cat, columns=cat_column_names)
    X_num_df = pd.DataFrame(X_num, columns=num_column_names)
    y_gen_df = pd.DataFrame(y_gen, columns=y_column_names)
    if categorical:
        generated_df = pd.concat([X_num_df, X_cat_df, y_gen_df], axis=1)
        raw_train_df = pd.concat([train_X_num, train_X_cat, train_y], axis=1)
        raw_test_df = pd.concat([test_X_num, test_X_cat, test_y], axis=1)
        raw_val_df = pd.concat([val_X_num, val_X_cat, val_y], axis=1)
    else:
        generated_df = pd.concat([X_num_df, y_gen_df], axis=1)
        raw_train_df = pd.concat([train_X_num, train_y], axis=1)
        raw_test_df = pd.concat([test_X_num, test_y], axis=1)
        raw_val_df = pd.concat([val_X_num, val_y], axis=1)

    generated_df.to_csv(f'{save_dir}/generated_training_set.csv')
    raw_train_df.to_csv(f'{save_dir}/raw_training_set.csv')
    train_df = pd.concat([raw_train_df, generated_df], axis=0)
    train_df.to_csv(f'{save_dir}/combined_training_set.csv')

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
    # accuracy = model.score(test_X, test_Y)
    predictions = model.predict(test_X)
    predicted_probabilities = model.predict_proba(test_X)
    acc, roc_auc, f1, prec, rec, roc_curve_result = calc_metrics_updated(test_Y, predictions, predicted_probabilities, metrics=[balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve])
    # print(f'Accuracy: ',accuracy)
    print(f'OOF ACC: {np.mean(acc)}')
    print(f'OOF ROC AUC Score: {np.mean(roc_auc)}')
    print(f'OOF F1 Score: {np.mean(f1):.4f}')
    print(f'OOF Prec Score: {np.mean(prec):.4f}')
    print(f'OOF Recall Score: {np.mean(rec):.4f}')

    data = {
        'OOF Acc': [],
        'OOF ROC AUC': [],
        'OOF F1': [],
        'OOF Precision': [],
        'OOF Recall': []
    }
  
    data['OOF Acc'].append(f'{np.mean(acc):.4f}')
    data['OOF ROC AUC'].append(f'{np.mean(roc_auc):.4f}')
    data['OOF F1'].append(f'{np.mean(f1):.4f}')
    data['OOF Precision'].append(f'{np.mean(prec):.4f}')
    data['OOF Recall'].append(f'{np.mean(rec):.4f}')
    
    df = pd.DataFrame(data)
    save_path = f'{save_dir}/model_metrics.xlsx'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_excel(save_path, index=False, engine='openpyxl')

if __name__ == "__main__":
    train_test_catboost('temp_experiments/cb_cardio', 'cardio', categorical=True)