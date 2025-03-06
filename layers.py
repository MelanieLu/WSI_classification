import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Lambda


def FeaturePreprocessingLayer(dense_size, dropout_rate=None, kernel_regularizer_rate=None):
    """
    Layers that preprocess the feature vectors before the scoring and pooling

    Args:
        dense_size (list of int): Dense layers sizes
        activation (str): Activation function for dense layers.
        dropout_rate (float or None): Dropout rate for dropout layers (or None if no dropout).
        kernel_regularizer_rate (float or None): Regularization rate for kernel weights (or None if no regularization).
    """
    layer = tf.keras.Sequential(name="descriptors_reducer")
    
    kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer_rate) if kernel_regularizer_rate is not None else None
        
    if dense_size is not None:
        for size in dense_size:
            layer.add(Dense(size, 'tanh', kernel_regularizer))
            if dropout_rate is not None:
                layer.add(Dropout(dropout_rate))
    return layer


def ClassificationLayers(num_classes, dense_sizes, activation, dropout_rate, kernel_regularizer_rate=None):
    """
    Create the last layers of the MIL network: the classification part.

    Args:
        num_labels (int): The number of output labels.
        dense_sizes (list of int): List of sizes for dense layers. Set the number of layers and their sizes
        activation (str): Activation function for dense layers.
        dropout_rate (float): Dropout rate for dropout layers (or None if no dropout).
        kernel_regularizer_rate (float or None): Regularization rate for kernel weights (or None if no regularization).

    Returns:
        tf.keras.Sequential: A sequential model for multi-class classification.
    """
    layer = tf.keras.Sequential(name="classification_layer")

    kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer_rate) if kernel_regularizer_rate is not None else None

    for size in dense_sizes:
        layer.add(Dense(size,
                        activation=activation,
                        kernel_regularizer=kernel_regularizer))

        if dropout_rate is not None:
            layer.add(Dropout(dropout_rate))

    layer.add(Dense(num_classes, activation='sigmoid'))
    return layer


def AttentionScoringLayer(dense_size, dropout_rate=None, kernel_regularizer_rate=None):
    """
    Args:
        dense_size (int): Dense layer sizes
        dropout_rate (float or None): Dropout rate for dropout layers (or None if no dropout).
        kernel_regularizer_rate (float or None): Regularization rate for kernel weights (or None if no regularization).
    """
    layer = tf.keras.Sequential(name="attention_scores")

    kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer_rate) if kernel_regularizer_rate is not None else None

    layer.add(Dense(dense_size, activation="tanh", kernel_regularizer=kernel_regularizer))

    if dropout_rate is not None:
        layer.add(Dropout(dropout_rate))

    layer.add(Dense(1))
    layer.add(Lambda(lambda ts: tf.squeeze(ts, axis=-1)))
    return layer


def ChowderScoringLayer(dense_size):
    """
    Args:
        dense_size (int): Dense layer size

    """
    layer = tf.keras.Sequential(name="chowder_scores")
    layer.add(Dense(dense_size, activation='sigmoid'))
    layer.add(Dense(1, activation='sigmoid'))
    layer.add(Lambda(lambda ts: tf.squeeze(ts, axis=-1)))
    return layer


class AttentionPooling(tf.keras.layers.Layer):
    """
    Output:
        slide_feature_vector : Feature Vector representing the slide (after pooling)
    """
    def __init__(self, **kwargs):
        super().__init__(name="attention_pooling")

    def call(self, inputs):
        feature_vectors, attention_scores = inputs
        softmaxed_scores = tf.nn.softmax(attention_scores)
        slide_feature_vector = softmaxed_scores[..., tf.newaxis] * feature_vectors
        slide_feature_vector = tf.reduce_sum(slide_feature_vector, axis=-2)
        return slide_feature_vector


class ChowderPooling(tf.keras.layers.Layer):
    """
    Args:
        R (int): The number of top and bottom tile scores to consider.
    Output:
        slide_feature_vector : Feature Vector representing the slide (after pooling)
    """
    def __init__(self, R, **kwargs):
        super().__init__(name="chowder_pooling")
        self._R = R

    def call(self, inputs):
        _, chowder_scores = inputs
        min_scores = tf.math.top_k(-chowder_scores, k=self._R).values * -1
        max_scores = tf.math.top_k(chowder_scores, k=self._R).values
        slide_feature_vector = tf.concat([max_scores, min_scores[:, ::-1]], axis=-1)
        return slide_feature_vector
