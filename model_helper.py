import tensorflow as tf

import numpy as np

def embedding_for_encoder_and_decoder(src_vocab_size, 
                                      src_embedding_size, 
                                      tgt_vocab_size, 
                                      tgt_embedding_size, 
                                      share):
    
    if share:  
        with tf.variable_scope('encoder_embedding'):
            encoder_embedding = tf.get_variable('enc_embed',
                [src_vocab_size, src_embedding_size], dtype=tf.float32)
        
        decoder_embedding = encoder_embedding
    
    # otherwise, make enc_embedding, dec_embedding
    else: 
        with tf.variable_scope('encoder_embedding'):
            encoder_embedding = tf.get_variable('enc_embed',
                [src_vocab_size, src_embedding_size], dtype=tf.float32)
        
        with tf.variable_scope('decoder_embedding'):
            decoder_embedding = tf.get_variable('dec_embed',
                [tgt_vocab_size, tgt_embedding_size], dtype=tf.float32)
            
    return encoder_embedding, decoder_embedding

def positional_encoding(sequence_length, d_model):
    # This encoded vector includes information about word sequence
    # and then use sine, cos function to have sequence
    encoded_vec = np.array(
        [pos / np.power(10000, 2*i/d_model) for pos in range(sequence_length) for i in range(d_model)])
    
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    encoded_vec = np.reshape(encoded_vec, [sequence_length, d_model])
    
    return tf.constant(encoded_vec, dtype=tf.float32)

def pos_encoding_for_src_and_tgt(src_seq_len, tgt_seq_len, d_model):
    # make source, target positional encoding
    src_pos_encoding = positional_encoding(
        src_seq_len, d_model)
    tgt_pos_encoding = positional_encoding(
        tgt_seq_len, d_model)
    
    return src_pos_encoding, tgt_pos_encoding
        
def scaled_dot_product_attention(query, key, value, mask=False):
    key_size = float(key.get_shape().as_list()[-1])
    # transpose key_tensor for matmul
    key_transpose = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key_transpose) / tf.sqrt(key_size)
    
    # masking for decoder, encoder doesn't use mask
    if mask:
        temp_tensor = tf.ones_like(outputs[0, :, :])
        temp_tensor = tf.linalg.LinearOperatorLowerTriangular(temp_tensor).to_dense()
        temp_tensor = tf.expand_dims(temp_tensor, 0)
        
        masking = tf.tile(temp_tensor, [tf.shape(outputs)[0], 1, 1])
        negative_tensor = tf.ones_like(outputs) * (-2**32 + 1)
        outputs = tf.where(tf.equal(masking, 0), negative_tensor, outputs)
    
    outputs = tf.nn.softmax(outputs, axis=-1)
    # matmul outputs(attention_map) with value
    outputs = tf.matmul(outputs, value)
    
    return outputs
    
def multi_head_attention(query, key, value, d_model, num_heads, mask=False):
    query = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)(query)
    key = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)(key)
    value = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)(value)
    
    # query, key, value shape = [batch_size, sequence_length, d_model]
    query_shape = query.get_shape().as_list()
    
    # split the tensors for multi head attention
    query = tf.concat(tf.split(query, num_heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, num_heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, num_heads, axis=-1), axis=0)
    
    # query, key, value shape [batch_size*num_heads, sequence_length, d_model/num_heads]
    outputs = scaled_dot_product_attention(query, key, value, mask)
    # reshape the tensors into original shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
    outputs = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)(outputs)
    
    # check outputs_shape [batch_size, sequence_length, d_model]
    assert query_shape == outputs.get_shape().as_list()
    
    return outputs
    
def feed_forward(inputs, feed_forward_dim):
    inputs_shape = inputs.get_shape().as_list()[-1]
    outputs = tf.keras.layers.Dense(feed_forward_dim, 
                                    activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(inputs_shape)(outputs)
    
    return outputs

def layer_norm(inputs, epsilon=1e-6):
    # inputs shape [batch_size, seq_len, d_model]
    inputs_shape = inputs.get_shape().as_list()[-1:]
    mean = tf.keras.backend.mean(inputs, axis=[-1], keepdims=True)
    std = tf.keras.backend.std(inputs, axis=[-1], keepdims=True)
    
    gamma = tf.Variable(tf.ones(inputs_shape), trainable=False)
    beta = tf.Variable(tf.zeros(inputs_shape), trainable=False)
    
    return gamma * (inputs - mean) / (std + epsilon) + beta

def build_encoder_module(inputs, 
                         d_model, 
                         num_heads, 
                         feed_forward_dim, 
                         enc_num, 
                         mode, 
                         keep_prob):
    
    with tf.variable_scope('encoder_%d' % enc_num):
        # depending on the mode, change keep_prob for dropout
        keep_prob = keep_prob if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        
        ## first sublayer
        # multi-head attention
        attention_outputs = multi_head_attention(
            query=inputs, 
            key=inputs, 
            value=inputs, 
            d_model=d_model, 
            num_heads=num_heads)
            
        # add, layer_norm
        outputs = layer_norm(inputs + tf.nn.dropout(attention_outputs, 
                                                    keep_prob))

        ## second sublayer
        # outputs, feed_forward
        feed_forward_outputs = feed_forward(
            inputs=outputs, feed_forward_dim=feed_forward_dim)
        
        # add, layer_norm
        outputs = layer_norm(outputs + tf.nn.dropout(feed_forward_outputs, 
                                                     keep_prob))
    
    return outputs

def build_encoder(num_enc_layers, 
                  inputs, 
                  d_model, 
                  num_heads, 
                  feed_forward_dim, 
                  base_gpu, 
                  num_gpus, 
                  mode, 
                  keep_prob):
    
    outputs = inputs
    for i in range(num_enc_layers):
        with tf.device(get_device_str((base_gpu+i), num_gpus)):
            outputs = build_encoder_module(
                inputs=outputs, 
                d_model=d_model, 
                num_heads=num_heads, 
                feed_forward_dim=feed_forward_dim,
                enc_num=i,
                mode=mode, 
                keep_prob=keep_prob)
        
    return outputs

def build_decoder_module(inputs, 
                         enc_outputs, 
                         d_model, 
                         num_heads, 
                         feed_forward_dim, 
                         dec_num, 
                         mode, 
                         keep_prob):
    
    with tf.variable_scope("decoder_%d" % dec_num):
        # depending on the mode, change keep_prob for dropout
        keep_prob = keep_prob if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        
        ## first sublayer
        # multi-head attention
        masked_attention_outputs = multi_head_attention(
            query=inputs, 
            key=inputs, 
            value=inputs, 
            d_model=d_model, 
            num_heads=num_heads, 
            mask=True)
            
        # add, layer_norm
        outputs = layer_norm(inputs + tf.nn.dropout(masked_attention_outputs, 
                                                    keep_prob))
            
        ## second sublayer
        # multi-head attention
        attention_outputs = multi_head_attention(
            query=outputs,
            key=enc_outputs, 
            value=enc_outputs, 
            d_model=d_model, 
            num_heads=num_heads)
            
        # add, layer_norm
        outputs = layer_norm(outputs + tf.nn.dropout(attention_outputs, 
                                                     keep_prob))

            
        ## third sublayer
        # outputs, feed_forward
        feed_forward_outputs = feed_forward(inputs=outputs, 
                                            feed_forward_dim=feed_forward_dim)
        # add, layer_norm
        outputs = layer_norm(outputs + tf.nn.dropout(feed_forward_outputs, 
                                                     keep_prob))
    
    return outputs

def build_decoder(num_dec_layers, 
                  inputs, 
                  enc_outputs, 
                  d_model, 
                  num_heads, 
                  feed_forward_dim, 
                  base_gpu, 
                  num_gpus, 
                  mode, 
                  keep_prob):
    
    outputs = inputs
    for i in range(num_dec_layers):
        with tf.device(get_device_str((base_gpu+i), num_gpus)):
            outputs = build_decoder_module(
                inputs=outputs, 
                enc_outputs=enc_outputs,
                d_model=d_model, 
                num_heads=num_heads, 
                feed_forward_dim=feed_forward_dim,
                dec_num=i,
                mode=mode, 
                keep_prob=keep_prob)
        
    return outputs
