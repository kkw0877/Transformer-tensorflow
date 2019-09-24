import os
import re

import numpy as np

import tensorflow as tf

import model
import model_helper
from utils import preprocess_util

def before_train(hparams, train_model, sess):
    # information about train mode
    info = {'cur_epoch' : 0}
    
    # run initializer before train 
    sess.run(train_model.iterator.initializer)
    return info

def get_model_creator(hparams):
    if hparams.model_name == 'transformer':
        return model.Transformer
    else:
        raise ValueError('Unknown model %s!' % hparams.model_name)
            
def update_info(info, outputs):
    # [optimizer, output_tuple(train_loss, train_summary, global_step)]
    _, output_tuple = outputs 
    
    train_summary = output_tuple.train_summary
    global_step = output_tuple.global_step

    return train_summary, global_step
    
def run_eval(eval_model, eval_sess, out_dir):
    # before run eval, load latest checkpoint file
    with eval_model.graph.as_default():
        loaded_eval_model, _ = model_helper.create_or_load_model(
            eval_model.model, out_dir, eval_sess)
        
        eval_sess.run(loaded_eval_model.iterator.initializer)
        output_tuple = loaded_eval_model.eval(eval_sess)

        eval_loss = output_tuple.eval_loss
        eval_accuracy = output_tuple.accuracy
        
        question_texts = output_tuple.question_texts
        origin_answers = output_tuple.origin_answers
        model_answers = output_tuple.model_answers
        
        # eval result
        print('eval_loss: {:.4f}, eval_accuracy: {:.2f}'.format(eval_loss, eval_accuracy))

        decode_func = np.vectorize(lambda x : x.decode('utf-8'))
        for question_text, origin_answer, model_answer in zip(question_texts[:5], origin_answers[:5], model_answers[:5]):
            question = decode_func(question_text)
            origin_a = decode_func(origin_answer)
            model_a = decode_func(model_answer)
            
            print('question text: {}'.format(question))
            print('answer text: {}'.format(origin_a))
            print('model text: {}'.format(model_a), end='\n\n')
    
            
def train(hparams):
    # set variables needed to train model
    out_dir = hparams.out_dir
    num_epochs = hparams.num_epochs
    epochs_per_eval = hparams.epochs_per_eval
    
    # session, model(train, eval)
    model_creator = get_model_creator(hparams)
    train_model = model_helper.create_train_model(hparams, model_creator)
    eval_model = model_helper.create_eval_model(hparams, model_creator)
    
    # config proto for sessions
    config_proto = tf.ConfigProto(
        log_device_placement=hparams.log_device_placement,
        allow_soft_placement=hparams.allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    
    # sessions. train, eval
    train_sess = tf.Session(
        config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(
        config=config_proto, graph=eval_model.graph)
    
    # summary_writer for train
    summary_name = 'train_log'
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), train_model.graph)
    
    # initialize vars, tables before train
    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, out_dir, train_sess)

    # set dict that has vars about train process
    info = before_train(hparams, loaded_train_model, train_sess)
        
    last_eval_epoch = info['cur_epoch']
    
    # train
    while info['cur_epoch'] <= num_epochs:
        try:
            outputs = loaded_train_model.train(train_sess)
            train_summary, global_step = update_info(info, outputs)
            
        # end of 1 epoch, and then initialize iterator to restart 
        except tf.errors.OutOfRangeError:
            info['cur_epoch'] += 1
            train_sess.run(train_model.iterator.initializer)
            continue
            
        # update summary 
        summary_writer.add_summary(train_summary, global_step)
        
        # eval
        if info['cur_epoch'] - last_eval_epoch >= epochs_per_eval:
            last_eval_epoch = info['cur_epoch']
            
            # save params
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(out_dir, 'chatbot.ckpt'),
                global_step=global_step)

            print('cur_epoch: {}, train_loss: {:.4f}, train_accuracy: {:.2f}'.format(
                info['cur_epoch'], outputs[1].train_loss, outputs[1].accuracy))
            run_eval(eval_model, eval_sess, out_dir)
            
    # before train done, save params
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, 'chatbot.ckpt'),
        global_step=global_step)
        
    print('train_done')
