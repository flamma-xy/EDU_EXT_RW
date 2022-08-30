EDU_EXT_RW
================
This code is for the paper [Extractive Elementary Discourse Units for Improving Abstractive Summarization](https://dl.acm.org/doi/abs/10.1145/3477495.3531916)
-----------------------------------
Some code from [Presum](https://github.com/nlpyang/PreSumm/tree/70b810e0f06d179022958dd35c1a3385fe87f28c)


Environment
===============
```
pip install requirement.txt
```

Data
========
Downloda the [data](https://drive.google.com/drive/folders/1wUqyH8bSLTbODBI3LW_w3xwmeLl6vj6r?usp=sharing) and put in ./EDU_EXT_RW/bert_data

Train
========
EDU-Extractor
------------
```

python src/EDUextractor.py -task ext 
                           -mode train 
                           -bert_data_path <PATH_OF_BERT_DATA> 
                           -ext_dropout 0.1 
                           -model_path <MODEL_SAVE_PATH> 
                           -lr 2e-3 
                           -visible_gpus 0 
                           -report_every 50 
                           -save_checkpoint_steps 1000 
                           -batch_size 3000 
                           -train_steps 50000 
                           -accum_count 2 
                           -log_file <LOG_FILES> 
                           -use_interval true 
                           -warmup_steps 10000 
                           -max_pos 512

```

EDU-Rewriter
--------------
```
python src/EDUrewriter

```
Evaluate Model
=============
```
python src/EDU_ext_then_rewrite
```
