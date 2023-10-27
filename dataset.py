import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import METHOD_DICT, BEAM_TYPE_DICT, L_H_DICT
from config import TRAIN_TEST_SPLIT_RATIO
import config


TRAIN_K_MEAN = None
TRAIN_K_STD = None


def transform_examples(examples):
    def transform_example(example):
        global TRAIN_K_MEAN, TRAIN_K_STD
        
        method_class, beam_type_class, l_h_class, k_value = example[0], example[1], example[2], example[3]
        method_class, beam_type_class, l_h_class = METHOD_DICT[method_class], BEAM_TYPE_DICT[beam_type_class], L_H_DICT[l_h_class]

        method_one_hot = torch.nn.functional.one_hot(torch.tensor(method_class), num_classes=len(METHOD_DICT.keys()))
        beam_type_one_hot = torch.nn.functional.one_hot(torch.tensor(beam_type_class), num_classes=len(BEAM_TYPE_DICT.keys()))
        l_h_one_hot = torch.nn.functional.one_hot(torch.tensor(l_h_class), num_classes=len(L_H_DICT.keys()))

        if TRAIN_K_MEAN == None:
            TRAIN_K_MEAN, TRAIN_K_STD = torch.load(config.K_NORMS_FILE)
        normalized_k = (k_value - TRAIN_K_MEAN.item()) / TRAIN_K_STD.item()
        
        test_tensor = torch.cat((method_one_hot, beam_type_one_hot, l_h_one_hot, torch.tensor([normalized_k])), dim=0).type(torch.float32)
        
        return test_tensor
  
    transformed_examples = [transform_example(example) for example in examples]
    transformed_examples = torch.stack(transformed_examples)
    return transformed_examples


def load_df_and_map_categoricals(data_file):
    df = pd.read_csv(data_file, sep=',')
    df = df[["Method", "Beam Type", "L/h", "k", "result"]]
    df = df.drop_duplicates(subset=["Method", "Beam Type", "L/h", "k"])

    # shuffling the dataframe
    df = df.sample(frac=1)

    # preprocessing the data
    df["Method"] = df["Method"].map(METHOD_DICT)
    df["Beam Type"] = df["Beam Type"].map(BEAM_TYPE_DICT)
    df["L/h"] = df["L/h"].map(L_H_DICT)

    return df


def df_to_train_and_test_datasets(df):
    global TRAIN_K_MEAN, TRAIN_K_STD

    input_data = df[["Method", "Beam Type", "L/h", "k"]].to_numpy()
    target_data = df["result"].to_numpy()
    input_tensor = torch.tensor(input_data)

    method_one_hot = torch.nn.functional.one_hot(input_tensor[:,0].type(torch.int64), num_classes=len(METHOD_DICT.keys()))
    beam_type_one_hot = torch.nn.functional.one_hot(input_tensor[:,1].type(torch.int64), num_classes=len(BEAM_TYPE_DICT.keys()))
    l_h_one_hot = torch.nn.functional.one_hot(input_tensor[:,2].type(torch.int64), num_classes=len(L_H_DICT.keys()))

    X = torch.cat((method_one_hot, beam_type_one_hot, l_h_one_hot, input_tensor[:,3].unsqueeze(1)), dim=1).type(torch.float32)
    y = torch.tensor(target_data).unsqueeze(1).type(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=1)
    X_train, X_test, y_train, y_test = X_train.float(), X_test.float(), y_train.float(), y_test.float()

    train_k_mean, train_k_std = torch.mean(X_train[:, -1]), torch.std(X_train[:, -1])
    TRAIN_K_MEAN, TRAIN_K_STD = train_k_mean, train_k_std
    torch.save(torch.tensor([TRAIN_K_MEAN, TRAIN_K_STD]), config.K_NORMS_FILE)

    X_train[:, -1] = (X_train[:, -1] - train_k_mean) / train_k_std
    X_test[:, -1] = (X_test[:, -1] - train_k_mean) / train_k_std

    training_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return training_dataset, test_dataset


def dataloader_from_datasets(training_dataset, test_dataset, batch_size=config.BATCH_SIZE, shuffle=True):
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return training_dataloader, test_dataloader


def get_dataloaders(data_file):
    df = load_df_and_map_categoricals(data_file)
    datasets = df_to_train_and_test_datasets(df)
    return dataloader_from_datasets(datasets[0], datasets[1])