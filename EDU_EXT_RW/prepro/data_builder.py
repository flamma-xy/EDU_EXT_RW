import gc
import glob
import argparse
import logging
import os
import os.path as path
import torch
from multiprocess import Pool, Manager
from rouge import Rouge
 
import numpy as np

from others.logging import logger
from transformers import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from utils import _get_word_ngrams

from nltk.tokenize.treebank import TreebankWordDetokenizer
import xml.etree.ElementTree as ET


logger = logging.getLogger()



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' Algorithm for matching closest sentence in article for each summary sentence
'''
def match_by_rouge12(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"]) / 2 for score in scores]
        res.append(recalls)
    return res

def match_by_rougeL(article, abstract):
    rouge = Rouge(metrics=["rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [score["rouge-l"]["r"] for score in scores]
        res.append(recalls)
    return res

def match_by_rouge12L(article, abstract):
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    res = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [(score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3 for score in scores]
        res.append(recalls)
    return res

''' Extend current BERT data to include ROUGE-L recalls and oracle selection for each summay sentence
'''

def extend_to_guidabs(args):
    files = [fn for fn in glob.glob(args.bert_data_files)]
    sep_token = '[SEP]'
    cls_token = '[CLS]'
    pad_token = '[PAD]'
    tgt_bos = '[unused0]'
    tgt_eos = '[unused1]'
    tgt_sent_split = '[unused2]'
    a_lst = []
    for fn in files:
        real_name = path.basename(fn) ## path last part
        save_file = path.join(args.result_data_path, real_name)
        if (os.path.exists(save_file)):
            logger.info('Exist and ignore %s' % save_file)
            continue

        jobs = torch.load(fn)
        a_lst.append((real_name, args, jobs, save_file))

    pool = Pool(args.n_cpus)
    for _ in pool.imap(_extend_to_guidabs, a_lst):
        pass

    pool.close()
    pool.join()

def _extend_to_guidabs(params, scorer=None):
    real_name, args, jobs, save_file = params
    use_bert_basic_tokenizer=True

    logger.info('Processing %s' % real_name)
    datasets = []
    for d in jobs:
        # follow min_src_nsents/3: training, but not testing.
        # follow max_src_nsents/100: src_subtoken_idxs, sent_labels. src_txt is not truncated.
        # follow min_src_ntokens_per_sent/5: all fields, include src_text.
        # follow max_src_ntokens_per_sent/200: src_subtoken_idxs, cls_ids, segments_ids. src_txt is not truncated.
        # follow min_tgt_ntokens/5: tgt_subtoken_idxs for training, but not testing.
        # follow max_tgt_ntokens/500: tgt_subtoken_idxs
        'src'
        src_subtoken_idxs = d["src"]
        
        'tgt'
        model_name = 'bert-base-uncased'
        bert_model_path = './bert_pretrained/'
        tokenizer = BertTokenizer.from_pretrained(path.join(bert_model_path, model_name), do_lower_case=True)
        tgt = d["tgt_tok_list_list_str"]
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:args.max_tgt_ntokens]
        tgt_subtoken_idxs = tokenizer.convert_tokens_to_ids(tgt_subtoken)
        
        # 'src_sent_labels'
        src_sent_labels = d["labels"]

        # 'segs'
        segs = d["segs"]

        # 'clss'
        clss = d["clss"]

        # 'src_txt'
        sent_txt = d["sent_txt"]
        for i in range(len(sent_txt)):
            sent_txt[i] = TreebankWordDetokenizer().detokenize(sent_txt[i]) 
        
        # 'tgt_txt'
        tgt_txt = d["tgt_list_str"]
        tgt_txt = '<q>'.join(x for x in tgt_txt)
        

        # 'disco_src_txt'
        disco_src_txt = d["disco_txt"]
        disco_src_txt = [' '.join([''.join(tt) for tt in disco]) for disco in disco_src_txt ]
        
        
        # 'disco_labels'
        disco_labels = d["d_labels"]

        # 'disco_span'
        disco_span = d["d_span"]

        # 'disco_labels_idx'
        disco_labels_idx = []
        for idx, i in enumerate(disco_labels[0]):
            if i == 1:
                disco_labels_idx.append(idx)

        # 'disco_dep'
        disco_dep = d["disco_dep"]



        
        segments_ids = d["segs"]
        cls_ids = d["clss"]
        
        

        # make src_txt following max_src_nsents and max_src_ntokens_per_sent firstly
        sent_txt = [' '.join(sent.split()[:args.max_src_ntokens_per_sent]) for sent in sent_txt][:args.max_src_nsents]

        # verify consistency between data fields
        assert len(cls_ids) == len(sent_txt), "len of cls_ids %s, num of source sentences %s" % (len(cls_ids), len(sent_txt))
        unused_ids = [i for i, idx in enumerate(tgt_subtoken_idxs) if idx in [1, 3]]
        
        _tgt_txt = tgt_txt.split('<q>')
        
        
        if len(unused_ids) != len(_tgt_txt):
            logging.info("len of unused_ids %s, num of target sentences %s" % (len(unused_ids), len(tgt_txt)))
            _tgt_txt = _tgt_txt[:len(unused_ids)]

        # match oracle sentence for each summary
        
        try:
            # abs_art_scores = np.array(match_by_rougeL(src_txt, tgt_txt))
            abs_art_scores = np.array(match_by_rouge12L(disco_src_txt, _tgt_txt))
        except Exception as ex:
            # logger.warning("Ignore exception from match_by_rougeL: %s, len of src_txt %s, len of tgt_txt %s" % (ex, len(src_txt), len(tgt_txt)))
            logger.warning("Ignore exception from match_by_rouge12L: %s, len of src_txt %s, len of tgt_txt %s" % (ex, len(sent_txt), len(tgt_txt)))
            continue
        
        
        # 'abs_art_idx'
        
        abs_art_idx = np.argmax(abs_art_scores, axis=1)
        
        assert abs_art_scores.shape[0] == len(unused_ids)
        assert abs_art_scores.shape[1] == len(disco_src_txt)
        
        # 'src_tags'
        # generate guide tags for each summary sentence
        src_tags = np.zeros((len(src_subtoken_idxs), len(abs_art_idx)), dtype=np.int)
        for i, idx in enumerate(abs_art_idx):
            start = disco_span[idx][0] if disco_span[idx][0] -1 not in clss else disco_span[idx][0] - 1
            end = disco_span[idx][1] if disco_span[idx][1] + 1 not in clss  else disco_span[idx][1] + 1
            src_tags[start:end, i] = 1
        src_tags = src_tags.tolist()

        
        # 'tgt_tag'
        tgt_tags = np.zeros(len(tgt_subtoken_idxs), dtype=np.int)
        for i in range(len(unused_ids)):
            start = unused_ids[i]
            end = unused_ids[i + 1] if i + 1 < len(unused_ids) else len(tgt_subtoken_idxs)
            tgt_tags[start:end] = i + 1  # 0 is skipped for padding
        tgt_tags = tgt_tags.tolist()

        data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": src_sent_labels, "segs": segs, 'clss': clss,
                       "src_txt": sent_txt, "tgt_txt": tgt_txt,
                       "abs_art_idx": abs_art_idx, "src_tags": src_tags, "tgt_tags": tgt_tags,
                        "disco_span":disco_span, "disco_src_txt":disco_src_txt, "disco_labels":disco_labels, "disco_labels_idx":disco_labels_idx, "disco_dep":disco_dep}
        datasets.append(data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_files", default='./source_bert_data/test/*.bert.pt')
    parser.add_argument("-result_data_path", default='/home/s2010187/myData/bert_data/test')
    parser.add_argument("-n_cpus", default=8, type=int)
    parser.add_argument("-temp_dir", default='./temp/')
    parser.add_argument('-log_file', default='./logs/data_builder.log')
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-max_src_nsents', default=150, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)
    
    # adding GuidAbs oracle info into existing BERT data
    extend_to_guidabs(args)
    

