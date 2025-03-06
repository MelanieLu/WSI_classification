# WSI classification

## General

This repository implements WSI classification pipeline. 
The inputs of the models are already extracted features from the tiles. 
The outputs are:
- A .csv with the cross-validation results
- .png plots to keep track of training history and confusion matrices for each split.

Early Stopping monitoring the validation loss is used to stop the training. 

The final prediction on the Test data consist in performing the ensembling of the models trained on each folds of the cross-validation.
Ensembling is performed by averaging the probabilities of the models. 

## Run the Script

```sh
python main.py --config-path ./config.yaml

```

## Parameters

```sh
config-path : path to the YAML config file
```


## Files

- `main.py`: Run the cross-validation
- `config.yaml`: Contains the configuration keys
- `utils_training.py`: Contains the training and cross-validation functions
- `utils_eval.py`: Contains the evaluation function, the plot functions to obtain metrics on the cross-validation
- `utils_data.py`: Contains function to load and process the data
- `models.py`: Contains the model classes and the function to select the model according to the config
- `layers.py`: Contains the model layers
