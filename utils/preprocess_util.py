import re
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from konlpy.tag import Okt

# unk, sos, eos
SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<pad>']
UNK = '<unk>'
UNK_ID = 0

def _preprocess_text(text, split_text, okt, stop_words=None):
    # some characters(.,~!?ㅠ) is changed into empty string
    text = text.lower()
    text = re.sub('[^가-힣a-z\d ]', '', text) 
    
    # split text using white space
    if split_text == 'white space':
        pass
    # split text using morpheme
    elif split_text == 'morpheme':
        text = ' '.join(okt.morphs(text, stem=True))
    # split text using stem
    elif split_text == 'stem':
        text = ' '.join(okt.morphs(text))
    else:
        raise ValueError('Unknown split_text %s!' % split_text)
        
    return text

def _make_file(inputs, labels, name):
    # set file name 
    src_file = 'src_%s_file.txt' % name
    tgt_file = 'tgt_%s_file.txt' % name
    
    # make file
    for file_name, texts in zip([src_file, tgt_file], [inputs, labels]):
        with open(file_name, 'wt', encoding='utf-8') as file:
            for text in texts:
                file.write(text + '\n')
    
def make_src_tgt_file(dataset, random_seed, test_size=0.2):
    # divide entire dataset into train, eval and infer 
    train_x, eval_x, train_y, eval_y = train_test_split(
        dataset['Q'], dataset['A'], random_state=random_seed, 
        test_size=test_size, shuffle=True)
    eval_x, infer_x, eval_y, infer_y = train_test_split(
        eval_x, eval_y, random_state=random_seed, 
        test_size=0.5, shuffle=True)
    
    _make_file(train_x, train_y, name='train')
    _make_file(eval_x, eval_y, name='eval')
    _make_file(infer_x, infer_y, name='infer')

def make_vocab_file(text_list, file_name='src'):
    words_list = []
    for text in text_list:
        words_list.extend(text.split())
        
    # get total_vocab 
    vocab_set = set(words_list)
    vocab_size = len(vocab_set) + len(SPECIAL_TOKENS) 
    vocab_file = '%s_vocab.txt' % file_name
    
    # make vocab file using special tokens list, vocab_set
    with open(vocab_file, 'wt', encoding='utf-8') as file:
        for token in SPECIAL_TOKENS:
            file.write(token + '\n')
                
        for vocab in vocab_set:
            file.write(vocab + '\n')
    
    return vocab_file, vocab_size
    
def preprocess(file_name, share, split_text, random_seed, test_size):
    dataset = pd.read_csv(file_name, encoding='utf-8')
    
    okt = Okt()
    # preprocess question, answer column before train
    # 'Q' means question which is source data / text, split_text, okt
    dataset['Q'] = dataset['Q'].apply(
        lambda question : _preprocess_text(question, split_text, okt)) 
    # 'A' means answer which is target data 
    dataset['A'] = dataset['A'].apply(
        lambda answer : _preprocess_text(answer, split_text, okt)) 
    
    # make source, target files for train, eval and infer mode 
    make_src_tgt_file(dataset, random_seed, test_size)
    # divide dataset into question and answer
    question_list = dataset['Q'].tolist()
    answer_list = dataset['A'].tolist()
    
    # if share, share vocab_file and vocab_size 
    if share:
        src_vocab_file, src_vocab_size = make_vocab_file(question_list + answer_list)
        tgt_vocab_file, tgt_vocab_size = src_vocab_file, src_vocab_size
    # otherwise, make each vocab_file, get each vocab_size 
    else:
        src_vocab_file, src_vocab_size = make_vocab_file(question_list)
        tgt_vocab_file, tgt_vocab_size = make_vocab_file(answer_list, file_name='tgt')
    
    file_vocab_dict = {'src_train_file' : 'src_train_file.txt',
                       'tgt_train_file' : 'tgt_train_file.txt',
                       'src_eval_file' : 'src_eval_file.txt',
                       'tgt_eval_file' : 'tgt_eval_file.txt',
                       'src_infer_file' : 'src_infer_file.txt',
                       'src_vocab_file' : src_vocab_file,
                       'src_vocab_size' : src_vocab_size,
                       'tgt_vocab_file' : tgt_vocab_file,
                       'tgt_vocab_size' : tgt_vocab_size}
    
    return file_vocab_dict
