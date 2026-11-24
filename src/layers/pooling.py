"""Small layers for the classifier heads and the embedding extractor.

Keras 3 does not allow TensorFlow functions (tf.reduce_sum, tf.stack, ...) to be
applied to the symbolic tensors of a functional model while it is being built,
so those steps live inside layers. They run unchanged under Keras 2 (TF 2.14).

autocast=False keeps each computation in the dtype of its input, exactly as the
bare tf ops did: under a mixed-precision policy the encoder output stays bfloat16
here instead of being cast by the layer.
"""
import tensorflow as tf


class MaskedMean(tf.keras.layers.Layer):
    """Mean of x over `axis`, counting only steps where mask is 1.

    inputs = [x, mask]. mask is broadcast against x; if it has fewer dimensions,
    leading axes are added (a (batch, steps, 1) mask against a stacked
    (layers, batch, steps, dim) x). A sample with no valid step gives 0.
    """
    def __init__(self, axis=1, **kwargs):
        kwargs.setdefault('autocast', False)
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        x, mask = inputs
        mask = tf.cast(mask, x.dtype)
        for _ in range(len(x.shape) - len(mask.shape)):
            mask = tf.expand_dims(mask, 0)
        total = tf.reduce_sum(tf.multiply(x, mask), axis=self.axis)
        return tf.math.divide_no_nan(total, tf.reduce_sum(mask, axis=self.axis))

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


class Stack(tf.keras.layers.Layer):
    """tf.stack of a list of tensors along a new `axis`."""
    def __init__(self, axis=0, **kwargs):
        kwargs.setdefault('autocast', False)
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(list(inputs), axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config
