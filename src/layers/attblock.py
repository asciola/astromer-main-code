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
                raise ValueError("latent_dim required when use_cache=True")
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

        if not self.use_cache:
            # Simple path for when caching is disabled (ignores kv_cache)
            # PRE-LN: Normalize BEFORE attention
            x_norm1 = self.layernorm1(x, training=training)
            attn_output, att_w, qk_v, (q_t, k_t, v_t) = self.mha(x_norm1, training=training, mask=mask)
            
            attn_output = self.dropout1(attn_output, training=training)
            # Add residual connection
            if self.use_leak:
                x = self.reshape_leak_1(x) + attn_output
            else:
                x = x + attn_output

            # PRE-LN: Normalize BEFORE FFN
            x_norm2 = self.layernorm2(x, training=training)
            ffn_output = self.ffn(x_norm2)
            ffn_output = self.dropout2(ffn_output, training=training)
            # Add residual connection
            if self.use_leak:
                ffn_output = self.reshape_leak_2(x) + ffn_output
            else:
                ffn_output = x + ffn_output

            if return_weights:
                return ffn_output, att_w, qk_v, (q_t, k_t, v_t), None
            return ffn_output, None

        # ---- Caching path (only traced if use_cache is True) ----
        # Cache is a single tensor: (B, cached_len, latent_dim)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # PRE-LN: Normalize input before attention
        x_norm1 = self.layernorm1(x, training=training)
        
        # Determine training status as a tensor
        is_training = tf.cast(training, tf.bool)
        can_use_cache = tf.cast(self.use_cache, tf.bool)

        # Normalize kv_cache to a Tensor (single tensor, not a tuple)
        l_dim = self.latent_dim if self.latent_dim else 1
        if kv_cache is None:
            kvc = tf.zeros((batch_size, 0, l_dim), dtype=x.dtype)
        else:
            kvc = kv_cache

        # Decision variables as Tensors
        has_cache_tensor = tf.greater(tf.shape(kvc)[1], 0)
        
        # Recurrent mode condition
        use_recurrent = tf.logical_and(
            tf.logical_and(can_use_cache, tf.logical_not(is_training)),
            tf.logical_or(tf.equal(seq_len, 1), has_cache_tensor)
        )

        def recurrent_path():
            x_q_norm = x_norm1[:, -1:, :]  # Use normalized input
            m_q = mask[:, -1:, :] if mask is not None else None
            
            def first_step():
                m_step = m_q[:, :, :1] if m_q is not None else None
                out, w, qk, qkv_tuple = self.mha(x_q_norm, mask=m_step, training=False)
                # Compute shared latent for cache
                c_kv_new = self.mha.compute_latent_kv(x_q_norm)  # (B, 1, latent_dim)
                return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], c_kv_new

            def update_step():
                out, w, qk, qkv_tuple = self.mha.attend_with_cached_kv(x_q_norm, kvc, m_q)
                # Compute shared latent for new token and append to cache
                c_kv_new = self.mha.compute_latent_kv(x_q_norm)  # (B, 1, latent_dim)
                # Cast cached values to match new values dtype for mixed precision
                kvc_cast = tf.cast(kvc, c_kv_new.dtype)
                nc = tf.concat([kvc_cast, c_kv_new], axis=1)  # (B, cached_len+1, latent_dim)
                return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], nc

            return tf.cond(has_cache_tensor, update_step, first_step)

        def full_sequence_path():
            out, w, qk, qkv_tuple = self.mha(x_norm1, mask=mask, training=training)
            
            def compute_prefill():
                return self.mha.compute_latent_kv(x_norm1)  # (B, L, latent_dim)

            def compute_empty():
                return tf.zeros((batch_size, 0, l_dim), dtype=x.dtype)
                
            nc = tf.cond(tf.logical_and(can_use_cache, tf.logical_not(is_training)),
                         compute_prefill, compute_empty)
            return out, w, qk, qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], nc

        # Execute branches
        attn_out, att_w, qk_v, q_t, k_t, v_t, n_cache = tf.cond(
            use_recurrent, recurrent_path, full_sequence_path)

        # PRE-LN: Residuals and Post-processing
        attn_out = self.dropout1(attn_out, training=training)

        # Add residual connection after attention
        if self.use_leak:
            leak_x = tf.cond(use_recurrent, lambda: x[:, -1:, :], lambda: x)
            x_after_attn = self.reshape_leak_1(leak_x) + attn_out
        else:
            x_after_attn = tf.cond(use_recurrent, lambda: x[:, -1:, :] + attn_out, lambda: x + attn_out)

        # PRE-LN: Normalize BEFORE FFN
        ffn_input = self.layernorm2(x_after_attn, training=training)
        ffn_out = self.ffn(ffn_input)
        ffn_out = self.dropout2(ffn_out, training=training)

        # Add residual connection after FFN
        if self.use_leak:
            ffn_out = self.reshape_leak_2(x_after_attn) + ffn_out
        else:
            ffn_out = x_after_attn + ffn_out
        
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
