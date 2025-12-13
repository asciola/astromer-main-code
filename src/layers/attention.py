import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask, m_alpha, mask_format='QK', temperature=0.):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if temperature != 0.:
        scaled_attention_logits = scaled_attention_logits * (1/temperature)
    
    qk_values = scaled_attention_logits
    if mask_format == 'K':
        steps = tf.shape(scaled_attention_logits)[2]
        mask_rshp = tf.tile(mask, [1,1,steps])
        mask_rshp = tf.transpose(mask_rshp, [0,2,1])
        mask_rshp = tf.minimum(1., mask_rshp)
        mask_rshp = tf.expand_dims(mask_rshp, 1)
        # selfattmask = tf.eye(steps) # Avoid to put attention on the same observation
        # mask_rshp  += selfattmask
        scaled_attention_logits += (mask_rshp*m_alpha)
        
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)

    if mask_format == 'Q':
        steps = tf.shape(scaled_attention_logits)[2]
        mask_rshp = tf.tile(mask, [1,1,steps])
        mask_rshp = tf.transpose(mask_rshp, [0,1,2])
        mask_rshp = tf.minimum(1., mask_rshp)
        mask_rshp = tf.expand_dims(mask_rshp, 1)
        scaled_attention_logits += (mask_rshp*m_alpha)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
        
    if mask_format == 'QK':
        steps = tf.shape(scaled_attention_logits)[2]
        mask_rshp = tf.tile(mask, [1,1,steps])
        mask_rshp += tf.transpose(mask_rshp, [0,2,1])
        mask_rshp = tf.minimum(1., mask_rshp)
        mask_rshp = tf.expand_dims(mask_rshp, 1)
        scaled_attention_logits += mask_rshp*m_alpha            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
    
    if mask_format == 'tanh':
        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.keras.activations.tanh(scaled_attention_logits)
        
    if mask_format == 'logits':
        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = scaled_attention_logits
        
    output = tf.matmul(attention_weights, v, name='Z')  # (..., seq_len_q, depth_v)

    return output, attention_weights, qk_values

class HeadAttentionMulti(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, m_alpha, mask_format, temperature):
        # super(HeadAttentionMulti, self).__init__()
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.mask_format = mask_format
        self.m_alpha     = m_alpha
        self.d_model     = self.num_heads * self.head_dim
        self.depth       = self.d_model // self.num_heads # final dimension
        self.temp        = temperature
        self.wq = tf.keras.layers.Dense(self.d_model, name='WQ')
        self.wk = tf.keras.layers.Dense(self.d_model, name='WK')
        self.wv = tf.keras.layers.Dense(self.d_model, name='WV')
        self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

    def split_heads(self, x, batch_size, name='qkv'):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    def call(self, x, training, mask=None):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, name='Q')  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, name='K')  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, name='V')  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights, qk_values = scaled_dot_product_attention(q, k, v, 
                                                                        mask=mask,
                                                                        m_alpha=self.m_alpha,
                                                                        mask_format=self.mask_format,
                                                                        temperature=self.temp)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights, qk_values, (q,k,v)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
        }
        return {**base_config, **config}

class HeadAttentionMultiCached(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, m_alpha, mask_format, temperature):
        # super(HeadAttentionMultiCached, self).__init__()
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.mask_format = mask_format
        self.m_alpha     = m_alpha
        self.d_model     = self.num_heads * self.head_dim
        self.depth       = self.d_model // self.num_heads # final dimension
        self.temp        = temperature
        self.wq = tf.keras.layers.Dense(self.d_model, name='WQ')
        self.wk = tf.keras.layers.Dense(self.d_model, name='WK')
        self.wv = tf.keras.layers.Dense(self.d_model, name='WV')
        self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

    def split_heads(self, x, batch_size, name='qkv'):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    # previous has the k & v from earlier parts of the sequence
    def call(self, x, training, previous_kv, mask=None):
        batch_size = tf.shape(x)[0]

        q_new = self.wq(x[:, -1:, :])  # (batch_size, 1, d_model)
        k_new = self.wk(x[:, -1:, :])  # (batch_size, 1, d_model)
        v_new = self.wv(x[:, -1:, :])  # (batch_size, 1, d_model)

        if previous_kv is not None:
            k_prev, v_prev = previous_kv
            # Concatenate along the sequence dimension (axis=1)
            k_combined = tf.concat([k_prev, k_new], axis=1)
            v_combined = tf.concat([v_prev, v_new], axis=1)
        else:
            # If no cache exists (first step), the new K/V is the combined K/V
            k_combined = k_new
            v_combined = v_new
        q = self.split_heads(q_new, batch_size, name='Q')  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k_combined, batch_size, name='K')  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v_combined, batch_size, name='V')  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights, qk_values = scaled_dot_product_attention(q, k, v, 
                                                                        mask=mask,
                                                                        m_alpha=self.m_alpha,
                                                                        mask_format=self.mask_format,
                                                                        temperature=self.temp)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights, qk_values, (q,k,v), (k_combined, v_combined)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
        }
        return {**base_config, **config}

class SimpleHeadAttentionMultiLatent(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, latent_dim, m_alpha, mask_format, temperature):
        # super(SimpleHeadAttentionMultiLatent, self).__init__()
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.latent_dim = latent_dim
        self.latent_head_dim  = latent_dim // num_heads
        self.mask_format = mask_format
        self.m_alpha     = m_alpha
        self.d_model     = self.num_heads * self.head_dim
        self.depth       = self.d_model // self.num_heads # final dimension
        self.temp        = temperature
        # Query projection (d_model)
        self.wq = tf.keras.layers.Dense(self.d_model, name='WQ')

        # K/V compression (d_model -> latent_dim)
        # Output to be cached.
        self.w_ckv = tf.keras.layers.Dense(self.latent_dim, use_bias=False, name='W_CKV')

        # K/V decompression (latent_dim -> d_model)
        # Needed for masking
        self.wuk = tf.keras.layers.Dense(self.d_model, use_bias=False, name='WUK') 
        self.wuv = tf.keras.layers.Dense(self.d_model, use_bias=False, name='WUV')

        self.wuk.build((None, self.latent_dim))
        self.wuv.build((None, self.latent_dim))

        # Final projection.
        self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

    def split_heads(self, x, batch_size, name='qkv'):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    def split_heads_latent(self, x, batch_size, name='latent'):
        """Split the last dimension (latent_dim) into (num_heads, latent_head_dim).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, latent_head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.latent_head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    def call(self, x, training, previous_kv=None, mask=None):
        batch_size = tf.shape(x)[0]

        if training:
            # Process full sequence
            q_full = self.wq(x) # (batch_size, L, d_model)
            q_latent = tf.matmul(q_full, self.wuk.kernel, transpose_b=True)
            combined = self.w_ckv(x) # C_cached: (batch_size, L, latent_dim)
        else:
            q_new = self.wq(x[:, -1:, :])  # (batch_size, 1, d_model)
            q_latent = tf.matmul(q_new, self.wuk.kernel, transpose_b=True) # (batch_size, 1, latent_dim)
            c_new = self.w_ckv(x[:, -1:, :])  # (batch_size, 1, latent_dim)

            if previous_kv is not None:
                # Concatenate along the sequence dimension (axis=1)
                combined = tf.concat([previous_kv, c_new], axis=1)
            else:
                # If no cache exists (first step), the new K/V is the combined K/V
                combined = c_new
        
        q = self.split_heads_latent(q_latent, batch_size, name='Q_latent') # (batch_size, num_heads, seq_len_q, latent_head_dim)
        k_latent = self.split_heads_latent(combined, batch_size, name='K_latent') # (batch_size, num_heads, seq_len_k, latent_head_dim)
        v_full = tf.matmul(combined, self.wuv.kernel) # (batch_size, seq_len, d_model)
        v = self.split_heads(v_full, batch_size, name='V')  # (batch_size, num_heads, seq_len_v, depth)

      
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q=1, depth=depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights, qk_values = scaled_dot_product_attention(
            q, k_latent, v, # q and k are latent, v is full-rank
            mask=mask,
            m_alpha=self.m_alpha,
            mask_format=self.mask_format,
            temperature=self.temp)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q=1, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q=1, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q=1, d_model)
        
        return output, attention_weights, qk_values, (q,k_latent,v), combined

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "latent_dim": self.latent_dim 
        }
        return {**base_config, **config}

class HeadAttentionMultiLatent(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, latent_dim, m_alpha, mask_format, temperature, **kwargs):
        super().__init__(**kwargs)
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.mask_format = mask_format
        self.m_alpha     = m_alpha
        self.d_model     = self.num_heads * self.head_dim
        self.temp        = temperature
        self.latent_dim  = latent_dim # Cache vector size. Should be much smaller than d_model
        self.latent_head_dim = latent_dim // num_heads
        
        # Input token -> query in latent space
        self.wq_latent = tf.keras.layers.Dense(self.latent_dim, name='WQ_latent')

        # Input token -> new latent
        self.w_latent_embedding = tf.keras.layers.Dense(self.latent_dim, name='W_latent_embedding')

        # New Latent -> k/v_latent
        self.wk_latent = tf.keras.layers.Dense(self.latent_dim, use_bias=False, name='W_K_latent')
        self.wv_latent = tf.keras.layers.Dense(self.latent_dim, use_bias=False, name='W_V_latent')
        
        # latent context -> model space
        self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

    def split_heads_latent(self, x, batch_size, name='latent'):
        """Split the last dimension (latent_dim) into (num_heads, latent_head_dim).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, latent_head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.latent_head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    # previous_kv = None | (latent_k, latent_v)
    def call(self, x, training, previous_kv=None, mask=None):
        
        batch_size = tf.shape(x)[0]
        
        # Latent query for last token
        q_latent = self.wq_latent(x[:, -1:, :]) # (batch_size, 1, latent_dim)
        
        c_new = self.w_latent_embedding(x[:, -1:, :]) # (batch_size, 1, latent_dim)

        # latent k/v for last token
        k_new_latent = self.wk_latent(c_new) # (batch_size, 1, latent_dim)
        v_new_latent = self.wv_latent(c_new) # (batch_size, 1, latent_dim)

        if previous_kv is not None:
            k_combined = tf.concat([previous_kv[0], k_new_latent], axis=1) # (batch_size, L, latent_dim)
            v_combined = tf.concat([previous_kv[1], v_new_latent], axis=1)
        else:
            k_combined = k_new_latent
            v_combined = v_new_latent

        q = self.split_heads_latent(q_latent, batch_size, name='Q_latent') # (batch_size, num_heads, 1, latent_head_dim)
        k = self.split_heads_latent(k_combined, batch_size, name='K_latent') # (batch_size, num_heads, latent_cache_len, latent_head_dim)
        v = self.split_heads_latent(v_combined, batch_size, name='V_latent') # (batch_size, num_heads, latent_cache_len, latent_head_dim)
        
        # Adjust mask to match lengths of latent cache.
        latent_len = tf.shape(k)[2]
        if mask is not None:
            # Slice the mask to match the latent cache length
            mask_latent = mask[:, -latent_len:, :]   # (B, L_latent, 1)

            # Reshape to broadcast with attention logits (B, H, 1, L_latent)
            mask_latent = tf.reshape(mask_latent, (batch_size, 1, 1, latent_len))
        else:
            mask_latent = None

        scaled_attention, attention_weights, qk_values = scaled_dot_product_attention(q, k, v, 
                                                                        mask=mask_latent,
                                                                        m_alpha=self.m_alpha,
                                                                        mask_format=self.mask_format,
                                                                        temperature=self.temp)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_latent_attention = tf.reshape(scaled_attention, (batch_size, 1, self.latent_dim))
        output = self.dense(concat_latent_attention) # (batch, 1, d_model)
        
        return output, attention_weights, qk_values, (q, k, v), (k_combined, v_combined)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim
        }
        return {**base_config, **config}