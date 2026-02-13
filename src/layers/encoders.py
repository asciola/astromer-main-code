import tensorflow as tf

from src.layers.attblock import AttentionBlock
from src.layers.positional import positional_encoding, PositionalEncoder2


class Encoder(tf.keras.Model):
    """ Encoder as it was defined in Astromer I """
    def __init__(self, 
                 window_size,
                 num_layers, 
                 num_heads, 
                 head_dim, 
                 mixer_size=128,
                 dropout=0.1, 
                 pe_base=1000, 
                 pe_dim=128,
                 pe_c=1., 
                 m_alpha=-0.5,
                 mask_format='Q',
                 use_leak=False,
                 temperature=0.,
                 use_cache=False,
                 latent_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        # super().__init__(**kwargs)

        self.window_size    = window_size
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.head_dim       = head_dim
        self.mixer_size     = mixer_size
        self.dropout        = dropout
        self.pe_base        = pe_base
        self.pe_c           = pe_c
        self.pe_dim         = pe_dim
        self.mask_format    = mask_format
        self.m_alpha        = m_alpha
        self.use_leak       = use_leak
        self.temp           = temperature
        self.use_cache      = use_cache
        self.latent_dim     = latent_dim

        if self.use_cache and self.latent_dim is None:
            raise ValueError("latent_dim must be provided when use_cache=True")

        self.kv_caches      = [None]*num_layers
        self.inp_transform  = tf.keras.layers.Dense(self.pe_dim, name='inp_transform')
        
        self.enc_layers = [AttentionBlock(self.head_dim, 
                                          self.num_heads, 
                                          self.mixer_size, 
                                          dropout=self.dropout, 
                                          mask_format=self.mask_format, 
                                          m_alpha=self.m_alpha,
                                          use_leak=self.use_leak,
                                          temperature=self.temp,
                                          use_cache=self.use_cache,
                                          latent_dim=self.latent_dim,
                                          name=f'att_layer_{i}')
                            for i in range(self.num_layers)]

        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.pe = PositionalEncoder2()
    def input_format(self, inputs):
        if 'seg_emb' in inputs.keys():
            window_size = self.window_size + 1 # if seg_emb exists then NSP is being applied
            x = tf.concat([inputs['input'], inputs['seg_emb']], axis=2, name='concat_mag_segemb')
        else:
            window_size = self.window_size
            x = inputs['input']

        x_transformed = self.inp_transform(x)           
        x_pe = self.pe(inputs['times'], 
                       d_model=self.pe_dim, 
                       base=self.pe_base, 
                       mjd=True, 
                       c=self.pe_c)
        # x_pe = positional_encoding(inputs['times'], 
        #                            self.pe_dim, 
        #                            base=self.pe_base, 
        #                            mjd=True, 
        #                            c=self.pe_c)
        x = x_transformed + x_pe   
        return x , window_size

    def output_transform(self, inputs):
        return inputs

    def reset_caches(self):
        self.kv_caches = [None] * self.num_layers

    def call(self, inputs, training=False, return_weights=False, z_by_layer=False):
        # adding embedding and position encoding.
        x, window_size = self.input_format(inputs)  
        x = self.dropout_layer(x, training=training)
        
        # Use a local copy to avoid leaking tensors into self.kv_caches during tracing
        current_caches = list(self.kv_caches)
        output_by_layer = []
        
        for i in range(self.num_layers):
            if return_weights:
                x, w, qkvalues, qkv, nc = self.enc_layers[i](
                    x, training=training, kv_cache=current_caches[i],
                    mask=inputs['mask_in'], 
                    return_weights=True)
            else:
                x, nc = self.enc_layers[i](
                    x, training=training, kv_cache=current_caches[i], 
                    mask=inputs['mask_in'])
            
            # Update local list for depth-wise caching
            current_caches[i] = nc
            
            # Optionally update persistent state if in eager mode
            # This avoids "out of scope" errors during Graph tracing
            if tf.executing_eagerly():
                self.kv_caches[i] = nc

            x = self.output_transform(x)
            output_by_layer.append(x)

        if return_weights:
            return x, w, qkvalues
        if z_by_layer:
            return output_by_layer
            
        return  x # (batch_size, input_seq_len, d_model)

class SkipEncoder(Encoder):
    def call(self, inputs, training=False):
        # adding embedding and position encoding.
        x, window_size = self.input_format(inputs)  
        x = self.dropout_layer(x, training=training)

        att_outputs = tf.TensorArray(dtype=x.dtype, 
                                     size=self.num_layers, 
                                     name='skip_att')
        for i in range(self.num_layers):
            x, _ =  self.enc_layers[i](x, training=training, mask=inputs['mask_in'])
            att_outputs = att_outputs.write(i, x)
        out = tf.reduce_mean(att_outputs.stack(), axis=0)
        return out  # (batch_size, input_seq_len, d_model)
