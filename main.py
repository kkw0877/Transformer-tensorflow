import os
import sys
import argparse

import tensorflow as tf

import train
from utils import preprocess_util

FLAGS = None

def add_arguments(parser):
    # Network
    parser.add_argument('--keep_prob', default=0.5, type=float)
    parser.add_argument('--model_name', default='transformer', type=str)
    parser.add_argument('--d_model', default=50, type=int)
    parser.add_argument('--num_heads', default=5, type=int)
    parser.add_argument('--feed_forward_dim', default=16, type=int)
    parser.add_argument('--num_enc_layers', default=2, type=int)
    parser.add_argument('--num_dec_layers', default=2, type=int)

    # Initializer
    parser.add_argument('--init_op', default='glorot_uniform', type=str,
                        help="glorot_normal | glorot_uniform | uniform")
    parser.add_argument('--init_weight', default=0.1, type=float)
    
    # Embedding
    parser.add_argument('--src_embedding_size', default=50, type=int)
    parser.add_argument('--tgt_embedding_size', default=50, type=int)
    
    # Preprocessing
    parser.add_argument('--file_name', default='ChatbotData.csv', type=str)
    parser.add_argument('--share', default=True, type=bool)
    parser.add_argument('--split_text', default='white space', type=str)
    parser.add_argument('--test_size', default=0.05, type=float)
    
    # Iterator
    parser.add_argument('--sos', default='<s>', type=str)
    parser.add_argument('--eos', default='</s>', type=str)
    parser.add_argument('--pad', default='<pad>', type=str)
    parser.add_argument('--src_max_len', default=8, type=int)
    parser.add_argument('--tgt_max_len', default=8, type=int)
    parser.add_argument('--reshuffle_each_iteration', default=True, type=bool)
    
    # Config proto
    parser.add_argument('--log_device_placement', default=False, type=bool)
    parser.add_argument('--allow_soft_placement', default=True, type=bool)
    
    # Learning
    parser.add_argument('--num_epochs', default=400, type=int)
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='adam | rmsprop | sgd')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    
    # else
    parser.add_argument('--random_seed', default=None) 
    parser.add_argument('--num_ckpts', default=5, type=int)
    parser.add_argument('--out_dir', default='train_result', type=str)
    
    parser.add_argument('--epochs_per_eval', default=2, type=int)
    parser.add_argument('--epochs_per_infer', default=2, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)    

def create_hparams(flags):
    return tf.contrib.training.HParams(
        # Network
        keep_prob=flags.keep_prob,
        model_name=flags.model_name,
        d_model=flags.d_model,
        num_heads=flags.num_heads,
        feed_forward_dim=flags.feed_forward_dim,
        num_enc_layers=flags.num_enc_layers,
        num_dec_layers=flags.num_dec_layers,

        # Initializer
        init_op = flags.init_op,
        init_weight = flags.init_weight,
        
        # Embedding
        src_embedding_size=flags.src_embedding_size,
        tgt_embedding_size=flags.tgt_embedding_size,
    
        # Preprocessing 
        file_name=flags.file_name, 
        share=flags.share, 
        split_text=flags.split_text, 
        test_size=flags.test_size,
    
        # Iterator 
        sos=flags.sos, 
        eos=flags.eos,
        pad=flags.pad,
        src_max_len=flags.src_max_len, 
        tgt_max_len=flags.tgt_max_len, 
        reshuffle_each_iteration=flags.reshuffle_each_iteration,
    
        # Config proto 
        log_device_placement=flags.log_device_placement, 
        allow_soft_placement=flags.allow_soft_placement, 
    
        # Learning 
        num_epochs=flags.num_epochs, 
        optimizer=flags.optimizer, 
        learning_rate=flags.learning_rate, 
        
        # else
        random_seed=flags.random_seed, 
        num_ckpts=flags.num_ckpts, 
        out_dir=flags.out_dir, 
        epochs_per_eval=flags.epochs_per_eval, 
        epochs_per_infer=flags.epochs_per_infer, 
        num_gpus=flags.num_gpus, 
        batch_size=flags.batch_size)
    
def add_extra_arguments(hparams, extra_arguments):
    '''
     file_vocab_dict = {'src_train_file' : 'src_train_file.txt',
                        'tgt_train_file' : 'tgt_train_file.txt',
                        'src_eval_file' : 'src_eval_file.txt',
                        'tgt_eval_file' : 'tgt_eval_file.txt',
                        'src_infer_file' : 'src_infer_file.txt',
                        'src_vocab_file' : src_vocab_file,
                        'src_vocab_size' : src_vocab_size,
                        'tgt_vocab_file' : tgt_vocab_file,
                        'tgt_vocab_size' : tgt_vocab_size}
    '''
    
    for key, value in extra_arguments.items():
        name = '%s' % key
        hparams.add_hparam(name, value)

    return hparams
    
def main(unused_argv):
    default_hparams = create_hparams(FLAGS)

    # set vars before preprocess
    share = default_hparams.share
    file_name = default_hparams.file_name 
    split_text = default_hparams.split_text
    random_seed = default_hparams.random_seed
    test_size = default_hparams.test_size
    
    # get extra vars needed to train model
    file_vocab_dict = preprocess_util.preprocess(
        file_name, share, split_text, random_seed, test_size)
    hparams = add_extra_arguments(default_hparams, file_vocab_dict)

    # train model
    train.train(hparams)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
