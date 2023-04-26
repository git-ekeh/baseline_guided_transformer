#!/usr/bin/env python3
from keyword_oracle import TextRank4Keyword
import tensorflow as tf
import datasets
from datasets import load_dataset
import pandas as pd
import collections
import itertools


train_data = load_dataset("xsum", split="train")
sample_traindata = itertools.islice(train_data,0,50)
train_dict = {}
for example in sample_traindata:
    train_dict.update(example)


print(train_dict)
