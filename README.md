# baseline_guided_transformer

Transformer model built from scratch in TensorFlow

Fine-tuned on the task of summarization 

Inspiration for experiment: GSum: A General Framework for Guided Neural Abstractive Summarizations written by Dou et al. in 2021. The gitHub page for the model can be found here: https://github.com/neulab/guided_summarization. 

<img width="331" alt="transformer_guided" src="https://github.com/user-attachments/assets/a0b821a2-b88f-4b0f-9c37-2d14825eee7c">




Results (navigate into the Google Colab notebook and scroll to the bottom):

As you can see from the graph, the results of my summarizer were not the greatest. The summary largely consisted of the [CLS] token and one or two words repeated consistently. Despite these results, I realized that building a model from scratch with no pre-training strategy will always result in a model that will stray from accurate results. I did not use the masked langugae training technique or the causal language training technique. I provided a transformer built from scratch, with a little over 200,000 examples and guidance signals. Many open-source models use at least 5 times the amount of examples I have. Anyways, this was a fun experience, regardless of the results. If building from scratch use a pre-training strategy, either Masked Language Modelling or Causal Language Modelling

