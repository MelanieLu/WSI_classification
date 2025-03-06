import tensorflow as tf
import numpy as np
import yaml
from tqdm import tqdm
from datetime import datetime
import shutil
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold




def setup_env(config_path):
    config = load_config(config_path)

    # Set Up Environment Paths
    data_dir = Path(config.get("data_dir"))

    # Nb repeat
    repeat = config.get("repeat")

    # Create Save Directory (with timestamp)
    current_time = datetime.now()
    folder_name = current_time.strftime('%Y_%m_%d')
    save_dir = data_dir / folder_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"Folder '{save_dir}' created!")

    # Copy config file to save_dir
    shutil.copyfile("./config.yaml", save_dir / "config.yaml")

    return config, data_dir, save_dir, repeat


def load_metadata(config, data_dir):
    # choice of feature (from config)
    features_name = config.get('features')

    # load the training and testing features and metadata
    train_features_dir = data_dir / "train_input" / f"{features_name}_features"
    test_features_dir = data_dir / "test_input" / f"{features_name}_features"
    # SampleID and labels
    df_train = pd.read_csv(data_dir / "train_labels.csv")
    df_test = pd.read_csv(data_dir /  "test_labels.csv")

    return df_train, df_test, train_features_dir, test_features_dir


def load_config(path_config):
    """
    Load a YAML configuration file.

    Args:
        path_config (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(path_config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_train_data(df_train, feature_dir):
    """
    Load training data from files.

    Args:
        df_train (pd.DataFrame): DataFrame containing sample and label information from the training data
        feature_dir (str): Directory containing feature.

    Returns:
        np.ndarray, np.ndarray: Loaded features (X) and labels (y).
    """
    X = []
    y = []

    for sample, label in tqdm(df_train[["Sample ID", "Label"]].values):
        _features = np.load(feature_dir / sample)
        features = _features[:, :] # to adapt if features contains tile coordinates
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


def load_test_data(df_test, test_features_dir):
    """
    Load test data

    Args:
        df_test (pd.DataFrame): DataFrame containing metadata for the test set.
        test_features_dir (str): Directory containing feature files for the test samples.

    Returns:
        np.ndarray: Loaded test set features
    """
    X_test = []

    for sample in tqdm(df_test["Sample ID"].values):
        _features = np.load(test_features_dir / sample)
        features = _features[:, :] # to adapt if features contains tile coordinates
        X_test.append(features)

    X_test = np.array(X_test)
    return X_test


def assert_feature_shape(X, feature_name):
    shape_dict = {
                "moco": 2048,
                "UNI": 1536,
                "transpath": 768,
                "histossl": 768,
                }
    try:
        predefined_shape = shape_dict.get(feature_name)
        if predefined_shape is None:
            raise ValueError(f"Invalid Feature Name: {feature_name}")
        assert X.shape[-1] == predefined_shape, f"Expected shape {predefined_shape}, but got shape {X.shape[-1]}"
    except (ValueError, AssertionError) as e:
        raise AssertionError(f"Shape validation failed: {e}")



def create_splits(X, y, k=5, upsample=False):
    """
    Create k-fold splits of the dataset.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        k (int): Number of folds for cross-validation.
        upsample (bool): Whether to upsample positive samples in the training split.

    Returns:
        List of tuples: Each tuple contains (X_train, y_train, X_val, y_val) for a fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Shuffle ensures random splits
    splits = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if upsample:
            # Identify positive samples in the training set
            positiveID = np.where(y_train == 1)[0]
            # Upsample positive samples by repeating them
            X_train = np.concatenate([X_train, X_train[positiveID], X_train[positiveID]])
            y_train = np.concatenate([y_train, y_train[positiveID], y_train[positiveID]])

        splits.append((X_train, y_train, X_val, y_val))

    return splits



def data_generator(X, y, batch_size, is_training=True):
    """
    Create a data generator.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        batch_size (int): Batch size.
        is_training (bool): Whether the generator is used for training.

    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        dataset = dataset.shuffle(len(X)).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    
    return dataset

