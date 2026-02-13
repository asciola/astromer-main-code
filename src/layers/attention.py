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
    dk_sqrt = tf.cast(tf.math.sqrt(dk), matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / dk_sqrt
    if temperature != 0.:
        scaled_attention_logits = scaled_attention_logits * (1/temperature)
    
    qk_values = scaled_attention_logits
    if mask_format == 'K':
        if mask is not None:
            def tile_mask_k():
                is_single_col = tf.equal(tf.shape(mask)[2], 1)
                def do_tile():
                    steps = tf.shape(scaled_attention_logits)[2]
                    m_tile = tf.tile(mask, [1, 1, steps])
                    return tf.transpose(m_tile, [0, 2, 1])
                return tf.cond(is_single_col, do_tile, lambda: mask)

            mask_rshp = tf.cond(tf.equal(tf.rank(mask), 3), tile_mask_k, lambda: mask)
            mask_rshp = tf.minimum(1., mask_rshp)
            mask_rshp = tf.expand_dims(mask_rshp, 1) 
            scaled_attention_logits += (mask_rshp*m_alpha)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')

    if mask_format == 'Q':
        if mask is not None:
            def tile_mask_q():
                is_single_col = tf.equal(tf.shape(mask)[2], 1)
                def do_tile():
                    steps = tf.shape(scaled_attention_logits)[2]
                    return tf.tile(mask, [1, 1, steps])
                return tf.cond(is_single_col, do_tile, lambda: mask)

            mask_rshp = tf.cond(tf.equal(tf.rank(mask), 3), tile_mask_q, lambda: mask)
            mask_rshp = tf.minimum(1., mask_rshp)
            mask_rshp = tf.expand_dims(mask_rshp, 1)
            scaled_attention_logits += (mask_rshp*m_alpha)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')
        
    if mask_format == 'QK':
        if mask is not None:
            def tile_mask_qk():
                is_single_col = tf.equal(tf.shape(mask)[2], 1)
                def do_tile():
                    steps = tf.shape(scaled_attention_logits)[2]
                    return tf.tile(mask, [1, 1, steps])
                return tf.cond(is_single_col, do_tile, lambda: mask)

            mask_rshp = tf.cond(tf.equal(tf.rank(mask), 3), tile_mask_qk, lambda: mask)
            
            def symmetric_mask():
                m_s = tf.shape(mask_rshp)
                is_square = tf.equal(m_s[1], m_s[2])
                return tf.cond(is_square, 
                               lambda: mask_rshp + tf.transpose(mask_rshp, [0, 2, 1]),
                               lambda: mask_rshp)

            mask_rshp = symmetric_mask()
            mask_rshp = tf.minimum(1., mask_rshp)
            mask_rshp = tf.expand_dims(mask_rshp, 1)
            scaled_attention_logits += mask_rshp*m_alpha            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')
    
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

class SimpleHeadAttentionMultiLatent(tf.keras.layers.Layer):
    """
Multi-head latent attention with KV compression.
    
Architecture:
- Input (d_model) → Compressed K/V (latent_dim) [shared across heads]
- Q projected per-head to latent_dim for comparison
- V reconstructed from latent to full d_model
    
Memory savings: O(L×d_model) → O(L×latent_dim + H×L×latent_dim)
"""
    def __init__(self,
                 head_dim,
                 num_heads,
                 latent_dim,
                 m_alpha,
                 mask_format,
                 temperature,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim

        self.d_model = num_heads * head_dim
        self.depth = head_dim

        self.m_alpha = m_alpha
        self.mask_format = mask_format
        self.temperature = temperature

        # Projections
        self.wq = tf.keras.layers.Dense(self.num_heads * self.latent_dim, name="WQ")
        # Separate K compression (shared across heads)
        self.w_k = tf.keras.layers.Dense(self.latent_dim, use_bias=False, name="W_K")
        
        # Separate V path: compress then reconstruct
        self.w_v_compress = tf.keras.layers.Dense(self.latent_dim, use_bias=False, name="W_V_compress")
        self.w_v_reconstruct = tf.keras.layers.Dense(
            self.num_heads * self.head_dim, 
            use_bias=False, 
            name="W_V_reconstruct"
        )
        self.out_proj = tf.keras.layers.Dense(self.d_model, name="attmerge")

    def split_heads(self, x, batch_size, name='qkv'):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3], name=name)

    def split_heads_latent_query(self, x, batch_size, name='latent_q'):
        # x is (B, L, H*latent_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.latent_dim))
        return tf.transpose(x, [0, 2, 1, 3], name=name)

    def compute_latent_kv(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: tuple of (k_latent, v_latent) each (batch, seq_len, latent_dim)
        """
        k_latent = self.w_k(x)  # (B, L, latent_dim)
        v_latent = self.w_v_compress(x)  # (B, L, latent_dim)
        return k_latent, v_latent

    def call(self, x, mask=None, training=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size = tf.shape(x)[0]

        # ---- Q path: single projection to latent ----    
        q_latent_all = self.wq(x)  # (B, L, H * latent_dim)
        q = self.split_heads_latent_query(q_latent_all, batch_size, name='Q_latent') # (B, H, L, latent_dim)

        # ---- K path (compressed, shared across heads) ----
        k_latent_full = self.w_k(x)  # (B, L, latent_dim)
        k_latent = tf.expand_dims(k_latent_full, axis=1)  # (B, 1, L, latent_dim)

        # ---- V path (separate compression + reconstruction) ----
        v_compressed = self.w_v_compress(x)  # (B, L, latent_dim)
        v_latent_all = self.w_v_reconstruct(v_compressed)  # (B, L, H * head_dim)
        v = tf.reshape(v_latent_all, (batch_size, -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, [0, 2, 1, 3])  # (B, H, L, head_dim)

        tf.debugging.assert_equal(
            tf.shape(q)[1:], [self.num_heads, tf.shape(x)[1], self.latent_dim],
            message="Q shape mismatch"
        )
        tf.debugging.assert_equal(
            tf.shape(k_latent)[1:], [1, tf.shape(x)[1], self.latent_dim],
            message="K shape mismatch"
        )
        tf.debugging.assert_equal(
            tf.shape(v)[1:], [self.num_heads, tf.shape(x)[1], self.head_dim],
            message="V shape mismatch"
        )

        # ---- Attention ----
        # attn weights shape: (B, H, L_q, L_k)
        attn, attn_weights, qk_values = scaled_dot_product_attention(
            q, k_latent, v,
            mask=mask,
            m_alpha=self.m_alpha,
            mask_format=self.mask_format,
            temperature=self.temperature
        )

        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, self.d_model))

        output = self.out_proj(attn)
        
        return output, attn_weights, qk_values, (q, k_latent, v)

    def attend_with_cached_kv(self, x_q, kv_cache, mask=None):
        """
        x_q: (batch, 1, d_model)
        kv_cache: tuple of (k_cache, v_cache), each (batch, seq_len_k, latent_dim)
        """
        batch_size = tf.shape(x_q)[0]

        # Unpack cache
        k_cache, v_cache = kv_cache

        # Query
        q_latent_all = self.wq(x_q)            # (B, 1, H * latent_dim)
        q = self.split_heads_latent_query(q_latent_all, batch_size) # (B, H, 1, latent_dim)

        # Cached K (shared across heads)
        k_latent = tf.expand_dims(k_cache, axis=1)  # (B, 1, L_k, latent_dim)
        
        # Cached V (reconstruct from compressed)
        v_latent_all = self.w_v_reconstruct(v_cache)  # (B, L_k, H * head_dim)
        v = tf.reshape(v_latent_all, (batch_size, -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, [0, 2, 1, 3])  # (B, H, L_k, head_dim)

        attn, attn_weights, qk = scaled_dot_product_attention(
            q, k_latent, v,
            mask=mask,
            m_alpha=self.m_alpha,
            mask_format=self.mask_format,
            temperature=self.temperature
        )

        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, 1, self.d_model))

        output = self.out_proj(attn)
        return output, attn_weights, qk, (q, k_latent, v)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "latent_dim": self.latent_dim,
            "m_alpha": self.m_alpha,
            "mask_format": self.mask_format,
            "temperature": self.temperature 
        }
        return {**base_config, **config}
