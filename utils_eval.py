import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import itertools
from scipy.special import softmax
from utils_data import load_test_data



def evaluation(X_test, y_test, eval_model):
    """
    Evaluate a model's performance on a test set.

    Args:
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.
        eval_model (tf.keras.Model): Model to evaluate.

    Returns:
        np.ndarray, np.ndarray, float: Predicted scores, probabilities, and AUC score.
    """
    probabilities, scores = eval_model.predict(X_test, batch_size=8)
    scores = softmax(scores, axis=-1)
    auc = roc_auc_score(y_test, probabilities)
    return scores, probabilities, auc


def save_eval(aucs, save_dir, repeat):
    """
    Save evaluation results to a CSV file.

    Args:
        aucs (list): List of AUC scores.
        save_dir (str): Directory to save the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results.
    """
    df_eval = pd.DataFrame({"AUC": aucs})
    mean = df_eval.mean()
    std = df_eval.std()
    df_eval = df_eval.append(mean, ignore_index=True)
    df_eval = df_eval.append(std, ignore_index=True)
    df_eval.to_csv(save_dir / "evaluation.csv")
    return df_eval


def plot_history(history, path, split_id, mode='loss', ):
    """
    Plot training history.

    Args:
        history (tf.keras.callbacks.History): Training history.
        path (str): Directory to save the plot.
        split_id (str): ID for the split.
        mode (str): Metric to plot (e.g., 'loss').

    Returns:
        None
    """
    plt.plot(history.history[mode])
    plt.plot(history.history[f'val_{mode}'])
    plt.ylabel(f'{mode}')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path / f"history_{split_id}.png", facecolor="white", dpi=100)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, path, split_id):
    """
    Plot the confusion matrix.

    Parameters:
        y_true (numpy.ndarray): The true labels of the data.
        y_pred (numpy.ndarray): The predicted probabilities of the data.
        labels (list): List of class labels.

    Returns:
        None
    """

    # Convert predicted probabilities to predicted classes
    y_pred_classes = np.where(y_pred>0.5, 1, 0)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add the values in the cells of the matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="split",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path / f"conf_mat_{split_id}.png", facecolor="white", dpi=100)
    plt.close()
    
    plt.hist(y_pred.flatten())
    plt.title("Predictions")
    plt.xlim([0, 1])
    plt.savefig(path / f"prediction_distrib_{split_id}.png", facecolor="white", dpi=100)
    plt.close()

