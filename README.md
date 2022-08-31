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
Create a data folder ./EDU_EXT_RW/bert_data/, Downloda the [pre-processed data](https://drive.google.com/file/d/1Wtciys2lO39cvmC6J-gcUfnNYYIJyQUQ/view?usp=sharing) and put in ./EDU_EXT_RW/bert_data

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
python src/EDUrewriter.py

```
Evaluate Model
=============
Set the path of the best EDU-selector model and the best EDU-rewriter model in src/EDU_ext_then_rewrite.py
---------------------
```
python src/EDU_ext_then_rewrite.py
```
Citing
============
```
@inproceedings{xiong2022extractive,
  title={Extractive Elementary Discourse Units for Improving Abstractive Summarization},
  author={Xiong, Ye and Racharak, Teeradaj and Nguyen, Minh Le},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2675--2679},
  year={2022}
}
```
