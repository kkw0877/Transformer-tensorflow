import tensorflow as tf

import model_helper

from collections import namedtuple

class TrainOutput(namedtuple(
    'TrainOutput', ('train_loss', 'train_summary', 
                    'global_step', 'accuracy'))):
    
    pass

class EvalOutput(namedtuple(
    'EvalOutput', ('eval_loss', 'accuracy',
                   'question_texts', 'origin_answers', 'model_answers'))):
    
    pass

class InferOutput(namedtuple(
    'InferOutput', ('sample_texts'))):
    
    pass



class Transformer(object):
    def __init__(self, hparams, mode, iterator, reverse_target_vocab_table):
        # set parameters 
        self.set_params_initializer(hparams, mode, iterator)
        
        # build transformer graph
        outputs = self.build_graph(hparams)

        # set extra ops depending on mode
        self.set_train_or_infer(hparams, outputs, reverse_target_vocab_table)

        # set saver for save and restore
        self.saver = tf.train.Saver(tf.global_variables(), 
            max_to_keep=hparams.num_ckpts)
    
    def set_params_initializer(self, hparams, mode, iterator):
        # mode. train, eval and infer
        self.mode = mode
        # iterator for source, target data
        self.iterator = iterator
        
        # network
        self.d_model = hparams.d_model
        self.num_heads = hparams.num_heads
        self.feed_forward_dim = hparams.feed_forward_dim

        self.num_enc_layers = hparams.num_enc_layers
        self.num_dec_layers = hparams.num_dec_layers
        
        # dropout 
        self.keep_prob = hparams.keep_prob
        
        # vocab
        self.src_vocab_size = hparams.src_vocab_size
        self.src_embedding_size = hparams.src_embedding_size
        self.src_max_len = hparams.src_max_len
        
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.tgt_embedding_size = hparams.tgt_embedding_size
        self.tgt_max_len = hparams.tgt_max_len
        
        # share 
        self.share = hparams.share
        
        # number of gpus
        self.num_gpus = hparams.num_gpus

        # batch_norm
        self.batch_norm = hparams.batch_norm
        
        # global_step
        self.global_step = tf.Variable(0, trainable=False)

        # initializer
        initializer = model_helper.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        
        # embedding, positional encoding
        self.enc_embedding, self.dec_embedding = self.init_embeddings(hparams)
        self.src_pos_encoding, self.tgt_pos_encoding = self.pos_encoding(hparams)

        assert self.d_model == self.src_embedding_size
        assert self.d_model == self.tgt_embedding_size
        
    def init_embeddings(self, hparams):
        enc_embedding, dec_embedding = (
            model_helper.embedding_for_encoder_and_decoder(
                src_vocab_size=hparams.src_vocab_size, 
                src_embedding_size=hparams.src_embedding_size, 
                tgt_vocab_size=hparams.tgt_vocab_size, 
                tgt_embedding_size=hparams.tgt_embedding_size, 
                share=hparams.share))
        
        return enc_embedding, dec_embedding
    
    def pos_encoding(self, hparams):
        src_pos_encoding, tgt_pos_encoding = (
            model_helper.pos_encoding_for_src_and_tgt(
                src_seq_len=hparams.src_max_len, 
                tgt_seq_len=hparams.tgt_max_len, 
                d_model=hparams.d_model))
        
        return src_pos_encoding, tgt_pos_encoding
    
    def build_graph(self, hparams):
        # encoder outputs and decoder outputs
        with tf.variable_scope('transformer'):
            enc_outputs = self.build_encoder(hparams)
            dec_outputs = self.build_decoder(hparams, enc_outputs)
        
        # dense layer 
        with tf.variable_scope('output_layer'):
            self.output_layer = tf.layers.Dense(units=hparams.tgt_vocab_size)
            
            # labels, device_id
            labels = self.iterator.tgt_output
            if self.num_dec_layers < self.num_gpus: 
                device_id = self.num_dec_layers
            else: 
                device_id = self.num_dec_layers - 1
            
            with tf.device(model_helper.get_device_str(device_id, self.num_gpus)):
                logits = self.output_layer(dec_outputs) 
                ids = tf.keras.backend.argmax(logits, axis=-1)

                if self.mode != tf.estimator.ModeKeys.PREDICT: # train, eval
                    loss = self.compute_loss(hparams, logits, labels, dec_outputs)
                else: # predict(=infer)
                    loss = tf.constant(0.0) # unused
            
        return loss, logits, ids
    
    def build_encoder(self, hparams):
        ## encoder embedding input
        # source_input, embedding
        src_input = self.iterator.src_input
        enc_embed_input = tf.nn.embedding_lookup(self.enc_embedding, src_input)

        # enc_emb_inp = enc_embedding + src_pos_encoding
        enc_embed_input = enc_embed_input + self.src_pos_encoding
        
        with tf.variable_scope('encoder'):
            enc_outputs = self._build_encoder(hparams, enc_embed_input)
            
        return enc_outputs
    
    def build_decoder(self, hparams, enc_outputs):
        # target_output, embedding
        tgt_input = self.iterator.tgt_input
        seq_len = tgt_input.get_shape().as_list()[1]
        dec_embed_input = tf.nn.embedding_lookup(self.dec_embedding, tgt_input)

        # dec_emb_inp = dec_embedding + tgt_pos_encoding
        dec_embed_input = dec_embed_input + self.tgt_pos_encoding[:seq_len, :]
        
        with tf.variable_scope('decoder'):
            dec_outputs = self._build_decoder(hparams, dec_embed_input, enc_outputs)
            
        return dec_outputs

    def set_train_or_infer(self, hparams, outputs, reverse_target_vocab_table):
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.train_loss = outputs[0]
            ids = outputs[2]
            self.learning_rate = hparams.learning_rate
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            
            if hparams.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif hparams.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
            elif hparams.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            else:
                raise ValueError('Unknown Optimizer %s' % hparams.optimizer)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimizer = opt.minimize(
                    loss=self.train_loss,
                    var_list=params,
                    global_step=self.global_step)
                
            self.train_summary = self.create_train_summary()

        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.eval_loss = outputs[0]
            ids = outputs[2]

            # tensors for questions, original answers, model_output answers
            self.question_texts = reverse_target_vocab_table.lookup(
                tf.to_int64(self.iterator.src_input))
            self.origin_answers = reverse_target_vocab_table.lookup(
                tf.to_int64(self.iterator.tgt_output))
            self.model_answers = reverse_target_vocab_table.lookup(
                tf.to_int64(ids))
                
        else: # self.mode == tf.estimator.ModeKeys.PREDICT
            ids = outputs[2]
            self.sample_texts = reverse_target_vocab_table.lookup(
                tf.to_int64(ids))

        if self.mode != tf.estimator.ModeKeys.PREDICT: # train, eval
            equal = tf.equal(ids, tf.cast(self.iterator.tgt_output, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    def compute_loss(self, hparams, logits, labels, dec_outputs):
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), axis=1)
        
        loss = tf.reduce_mean(loss)
        
        return loss
        
    def _build_encoder(self, hparams, enc_embed_input, base_gpu=0):
        enc_outputs = model_helper.build_encoder(
            num_enc_layers=self.num_enc_layers, 
            inputs=enc_embed_input, 
            d_model=self.d_model, 
            num_heads=self.num_heads, 
            feed_forward_dim=self.feed_forward_dim, 
            base_gpu=base_gpu, 
            num_gpus=self.num_gpus, 
            mode=self.mode, 
            keep_prob=self.keep_prob,
            batch_norm=self.batch_norm,
            training=(self.mode == tf.estimator.ModeKeys.TRAIN))
        
        return enc_outputs
    
    def _build_decoder(self, hparams, dec_embed_input, enc_outputs, base_gpu=0):
        dec_outputs = model_helper.build_decoder(
            num_dec_layers=self.num_dec_layers, 
            inputs=dec_embed_input, 
            enc_outputs=enc_outputs, 
            d_model=self.d_model, 
            num_heads=self.num_heads, 
            feed_forward_dim=self.feed_forward_dim, 
            base_gpu=base_gpu, 
            num_gpus=self.num_gpus, 
            mode=self.mode, 
            keep_prob=self.keep_prob,
            batch_norm=self.batch_norm,
            training=(self.mode == tf.estimator.ModeKeys.TRAIN))
        
        return dec_outputs
    
    def create_train_summary(self):
        train_summary = tf.summary.merge(
            [tf.summary.scalar('train_loss', self.train_loss)])
        
        return train_summary
    
    def train(self, sess):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        output_tuple = TrainOutput(train_loss=self.train_loss,
                                   train_summary=self.train_summary,
                                   global_step=self.global_step,
                                   accuracy=self.accuracy)
        
        return sess.run([self.optimizer, output_tuple])
    
    def eval(self, sess):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        output_tuple = EvalOutput(eval_loss=self.eval_loss,
                                  accuracy=self.accuracy,
                                  question_texts=self.question_texts,
                                  origin_answers=self.origin_answers,
                                  model_answers=self.model_answers)
        
        return sess.run(output_tuple)
    
    def infer(self, sess):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        output_tuple = InferOutput(sample_texts=self.sample_texts)
        
        return sess.run(output_tuple)
