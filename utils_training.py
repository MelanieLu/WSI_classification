
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_eval import plot_history, plot_confusion_matrix, evaluation, save_eval
from utils_data import create_splits, data_generator
from models import CustomEarlyStopping


def run_training(split, save_dir, Model, model_params, upsample=False, batch_size=8, learning_rate=0.001, nb_epochs=2):
    """
    Train the Model.

    Parameters:
    X (numpy.ndarray): Input data for training.
    y (numpy.ndarray): Labels for training.
    save_dir (str): Directory where training results and plots will be saved.
    Model (tf.keras.Model): The model class to be used for training.
    model_params (dict): Dictionary of model hyperparameters.
    upsample (bool): Whether to perform upsampling for imbalanced data (default: False).
    batch_size (int): Batch size for training (default: 8).
    learning_rate (float): Learning rate for the optimizer (default: 0.001).
    nb_epochs (int): Number of training epochs (default: 2).

    Returns:
    tf.keras.Model: Trained model.
    float: AUC score for the validation split.
    """    
    
    split = X_train, y_train, X_val, y_val
    train_generator = data_generator(X_train, y_train, batch_size, is_training=True)
    val_generator = data_generator(X_val, y_val, batch_size, is_training=False)
    
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    tf.keras.backend.clear_session()

    # Create early stopping callback
    early_stopping = CustomEarlyStopping(
            monitor='val_loss',  
            patience=20,
            restore_best_weights=True, 
            verbose=2, 
            start_from_epoch=10,
       )
    
    
    # Create the model
    ws_model = Model(input_shape=(None, X_train.shape[1], X_train.shape[2]), **model_params)

    # Compile the model
    ws_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])
    
    ws_model.summary()
    
    # Train
    history = ws_model.fit(
                    train_generator,
                    epochs=nb_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[early_stopping]
                )

    # Save plot of training history
    plot_history(history, save_dir, split_id)
    
    # Evaluation on the split
    att_scores, preds, auc = evaluation(X_val, y_val, ws_model)
    
    plot_confusion_matrix(y_val, preds, [0, 1], save_dir, split_id)
    print(f'AUC split {split_id} = {auc}')
    return ws_model, auc 


def run_cross_val(X, y, df_train, save_dir, Model, model_params, repeat, training_params):
    """
    Perform k-fold cross-validation training.

    Parameters:
    X (numpy.ndarray): Input data for training.
    y (numpy.ndarray): Labels for training.
    df_train (pandas.DataFrame): Dataframe containing training metadata.
    save_dir (str): Directory where training results and evaluation metrics will be saved.
    Model (tf.keras.Model): The model class to be used for training.
    model_params (dict): Dictionary of model hyperparameters.
    training_params (dict): Dictionary of training hyperparameters.

    Returns:
    pandas.DataFrame: DataFrame containing evaluation metrics for each cross-validation split.
    list: List of trained deep learning models.
    """
    # Create the Split
    splits = create_splits(X, y, k=5, upsample=upsample)
    
    models_cv = []
    aucs = []
    for k in range(5):
        print(f"####### Fold {k} #######")
        for n in range(repeat):
            print(f"####### Repeat {n} #######")
            split = splits[k]
            ws_model, auc = run_training(split, save_dir, Model, model_params, **training_params)
            models_cv.append(ws_model) # to change with a predict.py
            aucs.append(auc)
 
    # Save Evaluation Metrics in save_dir
    df_eval = save_eval(aucs, save_dir, repeat)
    print(f"Auc CV (repeat {repeat} times) = {round(np.mean(aucs), 3)} +/- {round(np.std(aucs), 3)}")
    return df_eval, models_cv

