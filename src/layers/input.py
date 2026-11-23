import tensorflow as tf

from tensorflow.keras.layers import Layer, Softmax

class GammaWeight(Layer):
    """ Weighted trainable average between attention outputs and input embeddings """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.softmax_act = Softmax(name='softmax')
        
    def build(self, input_shape):
        initial_value = tf.ones([input_shape[0]]) * 1/input_shape[0]
        self.gamma = tf.Variable(initial_value=initial_value,
                                 trainable=True,
                                 dtype=tf.float32,
                                 name='gamma')
        
    def call(self, inputs, training):
        # Cast gamma to match input dtype
        gamma = tf.cast(self.gamma, inputs.dtype)
        gamma = self.softmax_act(tf.expand_dims(gamma, 0))
        gamma = tf.reshape(gamma, [tf.shape(inputs)[0], 1, 1])
        x = tf.multiply(inputs, gamma)
        x = tf.reduce_mean(x, axis=0)
        return x
    
class AddMSKToken(Layer):
    """ Create MSK token """
    def __init__(self, 
                 trainable=True,
                 window_size=200,
                 on=['input'],
                 **kwargs):

        super().__init__(**kwargs)

        self.trainable = trainable
        self.on = on
        self.window_size = window_size

    def build(self, input_shape):
        # add_weight rather than a bare tf.Variable, so the token is tracked as a
        # layer weight (trained and saved) under both Keras 2 and Keras 3.
        self.msk_token = self.add_weight(name='msk_token',
                                         shape=(1, 1),
                                         initializer='zeros',
                                         dtype='float32',
                                         trainable=self.trainable)

    def call(self, inputs):
        # Return a NEW dict. This layer used to write into `inputs` in place, which
        # also rewrote the caller's dict of Input tensors: the functional model then
        # took this layer's outputs as its inputs and silently left the layer out,
        # so masked steps reached the encoder with their true values.
        outputs = dict(inputs)
        compute_dtype = inputs['mask_in'].dtype
        # Cast to the input dtype (bfloat16 under mixed precision)
        msk_token = tf.cast(self.msk_token, compute_dtype)
        one = tf.cast(1., compute_dtype)
        for key in self.on:
            kept = tf.multiply(inputs[key], one - inputs['mask_in'])
            outputs[key] = tf.add(kept, tf.multiply(inputs['mask_in'], msk_token))
        return outputs


