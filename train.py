#!/usr/bin/env python3

import tensorflow as tf
import datasets
from datasets import load_dataset
import transfotmers
from transformers import BertTokenizerFast
from transformer import DataCollatorWithPadding
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from rouge import Rouge
from rouge_score import rouge_scorer
import numpy as np
from transformer import Transformer
from tensorflow.keras.metrics import Metric
from keyword_oracle import TextRank4Keyword


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token


batch_size = 16 # this is for full training
encoder_max_length = 512
decoder_max_length = 128
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=encoder_max_length)
    guidance = tokenizer(batch["guidance"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["guidance_inputs"] = guidance.input_ids
    batch["guidance_attention_mask"] = guidance.input_ids
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exctly to `decoder_input_ids`
    # We have to make sure that he PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

'''
Going to update the train, val_data, and test set
with guidance key, and keyword values

train_data['guidance'] = guidance_kw

document --->
            ---> Decoder: Dialated Sliding Window Attention
                ------>                                         ---> FeedForward ---> Linear Layer ---> softmax
            --->
guidance ---> Decoder: Dialated Sliding Window Attention
'''

train_data = load_dataset("xsum", split="train")
val_data = load_dataset("xsum", split="validation")
test_data = load_dataset("xsum", split="test")

sample_traindata = itertools.islice(train_data,0,2000)
sample_valdata = itertools.islice(val_data,0,200)
sample_testdata = itertools.islice(test_data,0,20)
train_dict = {}
val_dict = {}
test_dict = {}


def keywordExtraction(x):
    textrank = TextRank4Keyword()
    textrank.analyze(x, candidate_pos =['NOUN', 'PROPN'], window_size = 4, lower = False)
    guidance_kw = textrank.get_keywords()
    return guidance_kw

for count, example in enumerate(sample_traindata):
    x_sum_train_dict = {"id": train_data['id'][count],
                        "document": train_data['document'][count],
                        "summary": str(train_data['summary'][count]),
                        "guidance": keywordExtraction(train_data['document'][count])}
    train_dict.update({str(count): x_sum_train_dict})

for count, example in enumerate(sample_valdata):
    x_sum_val_dict = {"id": val_data['id'][count],
                        "document": val_data['document'][count],
                        "summary": str(val_data['summary'][count]),
                        "guidance": keywordExtraction(val_data['document'][count])}
    val_dict.update({str(count): x_sum_val_dict})

for count, example in enumerate(sample_testdata):
    x_sum_test_dict = {"id": test_data['id'][count],
                        "document": test_data['document'][count],
                        "summary": str(test_data['summary'][count]),
                        "guidance": keywordExtraction(test_data['document'][count])}
    test_dict.update({str(count): x_sum_test_dict})




class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):

        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)



# Since the target sequences are padded, it is important to apply a padding mask when calculating the loss. Use the cross-entropy loss function

def rouge_scores(predicted, target):
    evaluator = Rouge(metrics=['rouge-l'])

    scores = evaluator.get_scores(predicted, target)
'''
model = Transformer()
model.compile(optimizer=optimizer,
              loss = 'categorical_crossentropy',
              metrics=[Metric('rouge')])
'''
