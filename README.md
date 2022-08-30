EDU_EXT_RW
================
This code is for the paper [Extractive Elementary Discourse Units for Improving Abstractive Summarization](https://dl.acm.org/doi/abs/10.1145/3477495.3531916)
-----------------------------------

Data
========
Downloda the [data](https://drive.google.com/drive/folders/1wUqyH8bSLTbODBI3LW_w3xwmeLl6vj6r?usp=sharing) and put in ./EDU_EXT_RW/bert_data

Train
========
EDU-Extractor
------------
‘’‘python

python src/EDUextractor.py -task ext -mode train -bert_data_path /home/s2010187/MY/bert_data/ -ext_dropout 0.1 -model_path /home/s2010187/MY/models/ext2 -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm2 -use_interval true -warmup_steps 10000 -max_pos 512

‘’‘
