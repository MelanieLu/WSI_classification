import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from layers import ChowderScoringLayer, AttentionScoringLayer, ClassificationLayers, FeaturePreprocessingLayer
from layers import AttentionPooling, ChowderPooling



def choose_model(config):
    """
    Choose a model based on the config.

    Args:
        config (dict): A dictionary containing the config.

    Returns:
        tuple: A tuple containing the chosen model class and its associated parameters.
               If the model name is not recognized, returns (None, None).
    """
    model_name = config.get("model_name")

    if model_name == "Attention":
        Model = WSClassificationAttention
        model_params = {key: config[key] for key in ["dense_size_1", "dense_size_2",
                                                      "dropout_rate", "kernel_regularizer_rate"] if key in config}
    elif model_name == "Chowder":
        Model = WSClassificationChowder
        model_params = {key: config[key] for key in ["dense_size_1", "dense_size_2",
                                                      "dropout_rate", "R", "kernel_regularizer_rate"] if key in config}
    else:
        print("Model Unknown")
        Model = None
        model_params = None

    return Model, model_params


class CustomEarlyStopping(EarlyStopping):
    """
    Custom EarlyStopping callback with the option to start monitoring after a specific epoch.

    Args:
        start_from_epoch (int): The epoch from which monitoring should start.
    """
    def __init__(self, start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        super().on_epoch_end(epoch, logs)


class WSClassificationAttention(tf.keras.Model):
    def __init__(self, input_shape, dense_size_1=[256, 128], dense_size_2=[16, 8], dropout_rate=0.4, kernel_regularizer_rate=0.001):
        super().__init__()  # Fixed the superclass name to match the class name        
        self._process_feature_vector = FeaturePreprocessingLayer(dense_size=dense_size_1, dropout_rate=dropout_rate, kernel_regularizer_rate=kernel_regularizer_rate)
        self._tile_scoring = AttentionScoringLayer(dense_size=dense_size_1[-1], dropout_rate=None, kernel_regularizer_rate=kernel_regularizer_rate)
        self._tile_pooling = AttentionPooling()
        self._classif_layer = ClassificationLayers(num_classes=1, dense_sizes=dense_size_2, activation='relu', dropout_rate=0.4, kernel_regularizer_rate=kernel_regularizer_rate)
        self.build(input_shape)

    def call(self, inputs):
        x = self._process_feature_vector(inputs)
        scores = self._tile_scoring(x)
        slide_feature_vector = self._tile_pooling([x, scores])
        predictions = self._classif_layer(slide_feature_vector)
        return predictions, scores



class WSClassificationChowder(tf.keras.Model):
    def __init__(self, input_shape, dense_size_1=[256, 128], dense_size_2=[16, 8], dropout_rate=0.4, kernel_regularizer_rate=0.001, R=5):
        super().__init__()
        
        self._process_feature_vector = FeaturePreprocessingLayer(dense_size=dense_size_1, dropout_rate=dropout_rate, kernel_regularizer_rate=kernel_regularizer_rate)
        self._tile_scoring = ChowderScoringLayer(dense_size=dense_size_1[-1])
        self._tile_pooling = ChowderPooling(R=R)
        self._classif_layer = ClassificationLayers(num_classes=1, dense_sizes=dense_size_2, activation='relu', dropout_rate=dropout_rate, kernel_regularizer_rate=kernel_regularizer_rate)
        self.build(input_shape)
        
    def call(self, inputs):
        x = self._process_feature_vector(inputs)
        scores = self._tile_scoring(x)
        slide_feature_vector = self._tile_pooling([x, scores])
        predictions = self._classif_layer(slide_feature_vector)
        return predictions, scores

