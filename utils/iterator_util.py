import tensorflow as tf

from collections import namedtuple

class Iterator(
    namedtuple('Iterator', 
        ('initializer', 'src_input', 'tgt_input', 
         'tgt_output', 'src_seq_len', 'tgt_seq_len'))):
    
    pass

def get_iterator(src_dataset, 
                 tgt_dataset, 
                 src_vocab_table, 
                 tgt_vocab_table, 
                 batch_size, 
                 src_max_len, 
                 tgt_max_len,
                 sos, 
                 eos,
                 pad,
                 num_parallel_calls=4,
                 random_seed=None, 
                 reshuffle_each_iteration=True):
    
    # output_buffer_size for shuffle, prefetch
    output_buffer_size = batch_size * 1000
    
    src_pad = tf.cast(src_vocab_table.lookup(tf.constant(pad)), tf.int32)
    tgt_sos = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_pad = tf.cast(tgt_vocab_table.lookup(tf.constant(pad)), tf.int32)
    
    # src_dataset, tgt_dataset zip
    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # shuffle
    dataset = dataset.shuffle(output_buffer_size, random_seed, reshuffle_each_iteration)
    
    # split dataset
    dataset = dataset.map(lambda src, tgt : 
        (tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    
    # filter dataset 
    dataset = dataset.filter(lambda src, tgt : 
                             tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    
    # change words into ids
    dataset = dataset.map(lambda src, tgt:(tf.cast(src_vocab_table.lookup(src), tf.int32), 
                                           tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    
    # use sos, eos for target_input and target_output
    dataset = dataset.map(lambda src, tgt : (src,
                                             tf.concat([[tgt_sos], tgt], 0),
                                             tf.concat([tgt, [tgt_eos]], 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    
    
    # cut dataset to have same sequence length
    dataset = dataset.map(lambda src, tgt_in, tgt_out : (src[:src_max_len],
                                                         tgt_in, 
                                                         tgt_out),
                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        
    dataset = dataset.map(lambda src, tgt_in, tgt_out : (src,
                                                         tgt_in[:tgt_max_len], 
                                                         tgt_out[:tgt_max_len]),
                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        
    # add source, target sequence length
    dataset = dataset.map(lambda src, tgt_in, tgt_out: (src, 
                                                        tgt_in, 
                                                        tgt_out, 
                                                        tf.size(src), # source_sequence_length
                                                        tf.size(tgt_in)), # target_sequence_length
                         num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    
    # add pad to source, target dataset
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=(tf.TensorShape([src_max_len]),
                                                  tf.TensorShape([tgt_max_len]),
                                                  tf.TensorShape([tgt_max_len]),
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([])),
                                   padding_values=(src_pad,
                                                   tgt_pad,
                                                   tgt_pad,
                                                   0,
                                                   0))
    
    # initializer 
    iterator = dataset.make_initializable_iterator()
    initializer = iterator.initializer
    src_input, tgt_input, tgt_output, src_seq_len, tgt_seq_len = iterator.get_next()
    
    return Iterator(initializer=initializer, 
                    src_input=src_input, 
                    tgt_input=tgt_input, 
                    tgt_output=tgt_output,
                    src_seq_len=src_seq_len, 
                    tgt_seq_len=tgt_seq_len)
