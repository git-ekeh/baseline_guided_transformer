#!/usr/bin/env python3

#from keyword_oracle import TextRank4Keyword
import tensorflow as tf
import datasets
import transformers
from transformers import DataCollatorWithPadding
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
from keyword_oracle import TextRank4Keyword
from transformer import Transformer
import pandas as pd
import collections
import itertools


train_data = load_dataset("xsum", split="train")
val_data = load_dataset("xsum", split="validation")
test_data = load_dataset("xsum", split="test")

sample_traindata = itertools.islice(train_data,0,200)
sample_valdata = itertools.islice(val_data,0,20)
sample_testdata = itertools.islice(test_data,0,20)
train_dict = {}
val_dict = {}
test_dict = {}

def keywordExtraction(x):
    textrank = TextRank4Keyword()
    textrank.analyze(x, candidate_pos =['NOUN', 'PROPN'], window_size = 4, lower = False)
    guidance_kw = textrank.get_keywords()
    return guidance_kw

# condense into function
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
#print(test_dict)
#import ipdb; ipdb.set_trace()

# Convert dictionaries to dataframes and transpose them to get the appropriate features of the dataframes

df_train = pd.DataFrame.from_dict(train_dict)
df_val = pd.DataFrame.from_dict(val_dict)
df_test = pd.DataFrame.from_dict(test_dict)

transpose_df_train = df_train.T
transpose_df_val = df_val.T
transpose_df_test = df_test.T


# pickle the tensors in a separate file
# Convert the dataframe rows to lists to avoid error: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
transpose_df_train['id'] = transpose_df_train['id'].values.tolist()
transpose_df_train['document'] = transpose_df_train['document'].values.tolist()
transpose_df_train['summary'] = transpose_df_train['summary'].values.tolist()
transpose_df_train['guidance'] = transpose_df_train['guidance'].values.tolist()

# Convert Dataframes to Datasets
dataset_train = Dataset.from_pandas(transpose_df_train)
dataset_val = Dataset.from_pandas(transpose_df_val)
dataset_test = Dataset.from_pandas(transpose_df_test)

dataset_train = dataset_train.remove_columns("__index_level_0__")
dataset_train = dataset_train.remove_columns("id")
dataset_val = dataset_val.remove_columns("__index_level_0__")
dataset_val = dataset_val.remove_columns("id")
dataset_test = dataset_test.remove_columns("__index_level_0__")
dataset_test = dataset_test.remove_columns("id")


#Tokenization
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token


batch_size = 8 # this is for full training
encoder_max_length = 512
decoder_max_length = 128
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["document"],  padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)


    #Concatenate guidance strings using the [SEP] token
    guidance_text = ["[SEP]".join(guidance) for guidance in batch["guidance"]]
    guidance = tokenizer(guidance_text, padding="max_length", truncation=True, max_length=encoder_max_length)


    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["guidance_inputs"] = guidance.input_ids
    batch["guidance_attention_mask"] = guidance.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exctly to `decoder_input_ids`
    # We have to make sure that he PAD token is ignored
    batch["labels"] = [[tokenizer.pad_token_id if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

dataset_train = dataset_train.map(process_data_to_model_inputs,
                                  batched=True,
                                  batch_size = batch_size)
dataset_val = dataset_val.map(process_data_to_model_inputs,
                                  batched=True,
                                  batch_size = batch_size)

# Converting the recently tokenized data into a tensorflow dataset
dataset_train = dataset_train.to_tf_dataset(
    columns=["input_ids","attention_mask","guidance_inputs","guidance_attention_mask","decoder_input_ids", "decoder_attention_mask", "labels"],
    label_cols=["labels"],
    batch_size = 2,
    collate_fn=data_collator,
    shuffle=True
)

dataset_val = dataset_val.to_tf_dataset(
    columns=["input_ids","attention_mask","guidance_inputs","guidance_attention_mask","decoder_input_ids", "decoder_attention_mask", "labels"],
    label_cols=["labels"],
    batch_size = 2,
    collate_fn=data_collator,
    shuffle=True
)
'''
# computing the rouge score

rouge = datasets.load_metric("rouge")

def compute_metric(pred):
  labels_ids = pred.labels_ids
  pred_ids = pred.predictions

  # all unnceccary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_uds == -100] = tokenizer.pad_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_type=["rouge2"])["rouge2"].mid

  return {
      "rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }
'''
# Define Hyperparameters

num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 200000
target_vocab_size = 200000
dropout_rate = 0.1
epochs = 10

# Constructing the model

model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                    input_vocab_size=input_vocab_size, target_vocab_size = target_vocab_size,
                    dropout_rate=dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics) # can also use any keras loss function
import evaluate
import numpy as np


# Train your model with the dataset

history = model.fit(dataset_train, epochs=epochs, validation_data=dataset_val)
'''
# Get visualizations for the data

history_dict = history.history
'''
#This should provide a dictionary with keys like:
 #- 'loss'
 #- 'accuracy'
 #- 'val_loss'
 #- 'val_accuracy'
'''
import matplotlib.pyplot as plt


def plot_training_history(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
plot_training_history(history_dict)
'''
def generate_summary(source_text, guidance_keywords, tokenizer, transformer, max_length=50):
    '''
    To actually get summaries I need to feed in the document portion of the
    x sum data set and the guidance, which willl require some sort of indexing
    Generate a sumamry using the Transformer model with greedy decoding

    Args:
        source_text (str): The input text to be summarized
        guidance_text (str): The guidance keywords to be used
        tokenizer: The tokenizer used to preprocess the input text.
        transformer: The trained Transformer model.
        max_length (int): The maximum length of the generated summary

    Returns:
        str: The generated summary
    '''

    # Use tokenized source and guidance text
    input_tokens = tokenizer.encode(source_text, padding="max_length", truncation=True, max_length=encoder_max_length)
    input_tokens = tf.expand_dims(input_tokens, 0) # Add batch dimension

    guidance_text = "[SEP]".join(guidance_keywords)
    guidance_tokens = tokenizer(guidance_text, padding="max_length", truncation=True, max_length=encoder_max_length)
    guidance_input_tokens = tf.expand_dims(guidance_tokens.input_ids, 0)
    guidance_attention_mask = tf.expand_dims(guidance_tokens.attention_mask, 0)


    # Initialize the target sequence with the start token
    target_tokens = tf.expand_dims([tokenizer.cls_token_id], 0)

    for _ in range(max_length):
        # Get the model's predictions
        logits = model({"input_ids": input_tokens, "attention_mask": guidance_attention_mask, "guidance_inputs": guidance_input_tokens, "guidance_attention_mask": guidance_attention_mask, "decoder_input_ids": target_tokens})
        predictions = tf.argmax(logits, axis=-1)[:, -1:] # Get the last token
        #print("Logits:", logits)
        #print("Predictions:", predictions)

        # Concatenate the predicted token to the target sequence
        target_tokens = tf.concat([target_tokens, tf.cast(predictions, dtype=tf.int32)], axis=-1)

        # If the end token in predicted, break the loop
        if predictions[0,0] == tokenizer.sep_token_id:
            break
    # Decode the target tokens into text
    summary = tokenizer.decode(target_tokens.numpy()[0], skip_special_tokens=True)

    return summary

# Get the first example from the dataset

#print(dataset_train["input_ids"])

# Access the document and guidance features
#source_text = train_dict["document"]

source_text = (train_dict['0']['document'])
guidance_keywords = (train_dict['0']['guidance'])

'''Do not need the guidance tokens in evaluation, because they are fed in during training'''
#guidance_text = (train_dict['0']['guidance'])



# Generate the summary

summary = generate_summary(source_text,guidance_keywords, tokenizer, model, max_length=50)

print("Generated summary:", summary)
