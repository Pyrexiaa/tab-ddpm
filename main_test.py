import numpy as np
import pandas as pd
import os

def convert_csv_to_npy(df_path, save_path):
    parent_directory = os.path.dirname(save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    dataset = pd.read_csv(df_path)
    data = dataset.to_numpy()
    np.save(save_path, data)

def convert_npy_to_csv(npy_path, save_path):
    parent_directory = os.path.dirname(save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    array = np.load(npy_path)
    np.savetxt(save_path, array, delimiter=',', fmt='%s')

def get_number_unique_values_from_csv(file_path):
    df = pd.read_csv(file_path)

    unique_values_dict = {}
    unique_counts_dict = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        unique_values = df[column].unique()
        unique_values_dict[column] = unique_values
        unique_counts_dict[column] = len(unique_values)

    return unique_values_dict, unique_counts_dict

def extract_cat_num_columns(csv_file_path, cat_save_path, num_save_path, y_save_path, cat_columns, label):
    parent_directory = os.path.dirname(cat_save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    parent_directory = os.path.dirname(num_save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    df = pd.read_csv(csv_file_path)
    num_columns = df.columns.difference(cat_columns).difference([label])
    cat_df = df[cat_columns].astype(int)
    num_df = df[num_columns]
    y_df = df[label].astype(int)
    cat_df.to_csv(cat_save_path, index=False, header=False)
    num_df.to_csv(num_save_path, index=False, header=False)
    y_df.to_csv(y_save_path, index=False, header=False)

def convert_synthetic_data_to_csv(exp_dir):
    for file in os.listdir(exp_dir):
        if (file.endswith('npy')):
            file_name = file[:-4]
            read_file = np.load(f'{exp_dir}/{file}')
            convert_npy_to_csv(f'{exp_dir}/{file}', f'{exp_dir}/{file_name}.csv')

def main():
    # data = np.load('data/churn2/y_train.npy')
    # print(data)
    # convert_npy_to_csv('data/diabetes/X_num_train.npy', 'data/diabetes/npy_to_csv/X_num_train.csv')

    # extract_cat_num_columns('data/ppum/csv/train/accepted_train_set_0_cosine_100.csv', 'data/ppum/extracted_csv/X_cat_train.csv', 'data/ppum/extracted_csv/X_num_train.csv', 'data/ppum/extracted_csv/y_train.csv', ['gender'], 'sga')
    # extract_cat_num_columns('data/ppum/csv/test/accepted_test_set_0_cosine_100.csv', 'data/ppum/extracted_csv/X_cat_test.csv', 'data/ppum/extracted_csv/X_num_test.csv', 'data/ppum/extracted_csv/y_test.csv', ['gender'], 'sga')
    # extract_cat_num_columns('data/ppum/csv/validation/accepted_validation_set_0_cosine_100.csv', 'data/ppum/extracted_csv/X_cat_val.csv', 'data/ppum/extracted_csv/X_num_val.csv', 'data/ppum/extracted_csv/y_val.csv', ['gender'], 'sga')

    # convert_csv_to_npy('data/ppum/extracted_csv/X_cat_test.csv', 'data/ppum/X_cat_test.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/X_cat_train.csv', 'data/ppum/X_cat_train.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/X_cat_val.csv', 'data/ppum/X_cat_val.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/X_num_train.csv', 'data/ppum/X_num_train.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/X_num_test.csv', 'data/ppum/X_num_test.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/X_num_val.csv', 'data/ppum/X_num_val.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/y_train.csv', 'data/ppum/y_train.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/y_test.csv', 'data/ppum/y_test.npy')
    # convert_csv_to_npy('data/ppum/extracted_csv/y_val.csv', 'data/ppum/y_val.npy')

    # Convert exp dataset from npy to csv and compare with initial dataset
    convert_synthetic_data_to_csv('exp/ppum/ddpm_tune_best')
    file = pd.read_csv('exp/ppum/ddpm_tune_best/y_train.csv')
    print(file.iloc[:, 0].value_counts())

if __name__ == '__main__':
    main()