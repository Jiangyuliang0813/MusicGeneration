# 使用LSTM生成音乐

## 本代码是大神Valerio Velardo的pytorch实现
- tensorflow版本 https://github.com/musikalkemist/generating-melodies-with-rnn-lstm
-  数据来源于 https://kern.humdrum.org/

## requirement 
- torch 
- music21
- musecore
- torchinfo

## 使用顺序
- 1st preprocess.py   -> 数据预处理1 -> dataset 
- 2nd datasetbuilding.py -> 数据预处理2 -> file_dataset  mapping.json
- 3rd dataloader.py -> 数据集制作 
- 4th train.py -> 训练保存模型
- 5th test.py -> 模型推理 


