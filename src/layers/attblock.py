import tensorflow as tf
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

from src.layers.attention import HeadAttentionMulti, SimpleHeadAttentionMultiLatent


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, mixer_size, 
                 dropout=0.1, m_alpha=-0.5, mask_format='Q', 
                 use_leak=False, temperature=0., use_cache=False, latent_dim=None, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mixer_size = mixer_size
        self.dropout = dropout
        self.mask_format = mask_format
        self.m_alpha = m_alpha
        self.use_leak = use_leak
        self.temp = temperature
        self.use_cache = use_cache
        self.latent_dim = latent_dim
        if use_cache:
            if latent_dim is not None:
                self.mha = SimpleHeadAttentionMultiLatent(self.head_dim, self.num_heads, self.latent_dim, m_alpha=self.m_alpha, mask_format=mask_format, temperature=self.temp)
        else:
            self.mha = HeadAttentionMulti(self.head_dim, self.num_heads, m_alpha=self.m_alpha, mask_format=mask_format, temperature=self.temp)
        self.ffn = point_wise_feed_forward_network(self.num_heads*self.head_dim, 
                                                   self.mixer_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if use_leak:
            self.reshape_leak_1 = tf.keras.layers.Dense(self.head_dim*self.num_heads)
            self.reshape_leak_2 = tf.keras.layers.Dense(self.head_dim*self.num_heads)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=None, mask=None, kv_cache=None, return_weights=False):
        if training is None:
            training = tf.keras.backend.learning_phase()

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Determine training status as a tensor
        is_training = tf.cast(training, tf.bool)
        can_use_cache = tf.cast(self.use_cache, tf.bool)

        # Normalize kv_cache to a Tensor to avoid Graph/Literal mismatch issues
        if kv_cache is None:
             l_dim = self.latent_dim if self.latent_dim else 1
             kvc = tf.zeros((batch_size, 0, l_dim))
        else:
             kvc = kv_cache

        # Decision variables as Tensors
        has_cache_tensor = tf.greater(tf.shape(kvc)[1], 0)
        
        # Recurrent mode condition
        # We use recurrent mode ONLY if caching is enabled AND not training AND (seq_len == 1 OR we have a cache)
        use_recurrent = tf.logical_and(
            tf.logical_and(can_use_cache, tf.logical_not(is_training)),
            tf.logical_or(tf.equal(seq_len, 1), has_cache_tensor)
        )

        def recurrent_path():
            x_q = x[:, -1:, :]
            m_q = mask[:, -1:, :] if mask is not None else None
            
            def first_step():
                # No cache yet: initial attention call
                m_step = m_q[:, :, :1] if m_q is not None else None
                out, w, qk, qkv_tuple = self.mha(x_q, mask=m_step, training=False)
                nc = self.mha.compute_latent_kv(x_q)
                return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], nc

            def update_step():
                # Use provided cache
                out, w, qk, qkv_tuple = self.mha.attend_with_cached_kv(x_q, kvc, m_q)
                new_kv = self.mha.compute_latent_kv(x_q)
                nc = tf.concat([kvc, new_kv], axis=1)
                return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], nc

            return tf.cond(has_cache_tensor, update_step, first_step)

        def full_sequence_path():
            # Process full sequence (Training or Validation/Inference batch)
            out, w, qk, qkv_tuple = self.mha(x, mask=mask, training=training)
            
            # Prefill cache if and only if use_cache=True AND training=False (validation/batch inference)
            def compute_prefill():
                return self.mha.compute_latent_kv(x)
            
            def compute_empty():
                l_dim = self.latent_dim if self.latent_dim else 1
                return tf.zeros((batch_size, 0, l_dim))
                
            nc = tf.cond(tf.logical_and(can_use_cache, tf.logical_not(is_training)),
                         compute_prefill, compute_empty)
            return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], nc

        # Execute branches. Flatten outputs to avoid nested structure issues in Graph mode.
        attn_out, att_w, qk_v, q_t, k_t, v_t, n_cache = tf.cond(
            use_recurrent, recurrent_path, full_sequence_path)

        # Residuals and Post-processing
        attn_out = self.dropout1(attn_out, training=training)

        if self.use_leak:
            leak_x = tf.cond(use_recurrent, lambda: x[:, -1:, :], lambda: x)
            attn_out = self.reshape_leak_1(leak_x) + attn_out

        attn_out = self.layernorm1(attn_out, training=training)
        ffn_out  = self.ffn(attn_out)
        ffn_out  = self.dropout2(ffn_out, training=training)

        if self.use_leak:
            ffn_out = self.reshape_leak_2(attn_out) + ffn_out

        ffn_out = self.layernorm2(ffn_out, training=training)
        
        # Convert empty tensor cache back to None ONLY if self.use_cache is False globally
        # Otherwise, return the tensor to satisfy Graph mode requirements
        final_cache_val = n_cache if self.use_cache else None

        if return_weights:
            return ffn_out, att_w, qk_v, (q_t, k_t, v_t), final_cache_val

        return ffn_out, final_cache_val

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "mixer_size": self.mixer_size,
            "dropout": self.dropout,
            "mask_format": self.mask_format,
            "m_alpha": self.m_alpha,
            "use_leak": self.use_leak,
            "use_cache": self.use_cache,
            "latent_dim": self.latent_dim,
            "temperature": self.temp
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the block from its parameters
        return cls(**config)