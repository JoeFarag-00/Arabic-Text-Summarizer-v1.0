import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from nltk.stem import ISRIStemmer
from pathlib import Path

stemmer = ISRIStemmer()
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

pd.set_option('display.max_rows', None)

df = pd.read_csv("Dataset/Arabic_Books.csv", nrows=100)
print(df.head())

with open(Path("Stopwords/Stopwords_list.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

def preprocess(text):
    if isinstance(text,str):
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        tokens = [token for token in tokens if token not in arabic_stopwords]
        # tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
        tokens = [token for token in tokens if not token.isdigit()]
        tokens = [token for token in tokens if token]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''

batch_size = 8

texts = df['text'].tolist()
summaries = df['summary'].tolist()

text_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
summary_batches = [summaries[i:i+batch_size] for i in range(0, len(summaries), batch_size)]

processed_texts = []
processed_summaries = []

with open("Preprocessed_Texts/preprocessed_texts.txt", "w", encoding="utf-8") as text_file, \
        open("Preprocessed_Texts/preprocessed_summaries.txt", "w", encoding="utf-8") as summary_file:

    for i, (text_batch, summary_batch) in enumerate(zip(text_batches, summary_batches)):
        batch_texts = [preprocess(text) for text in text_batch]
        batch_summaries = [preprocess(summary) for summary in summary_batch]

        processed_texts.extend(batch_texts)
        processed_summaries.extend(batch_summaries)

        for j, text in enumerate(batch_texts):
            text_file.write(f"Text {i * batch_size + j + 1}: {text}\n")

        for j, summary in enumerate(batch_summaries):
            summary_file.write(f"Summary {i * batch_size + j + 1}: {summary}\n")

df['text'] = processed_texts[:len(df)]
df['summary'] = processed_summaries[:len(df)]

model = TFAutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")

train_size = int(0.8 * len(df))
train_texts = df['text'][:train_size].tolist()
train_summaries = df['summary'][:train_size].tolist()
test_texts = df['text'][train_size:].tolist()
test_summaries = df['summary'][train_size:].tolist()

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.MeanSquaredError()

def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)[0]
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(texts, summaries):
    rouge = tf.keras.metrics.Mean()
    for i in range(len(texts)):
        inputs = tokenizer.encode(texts[i], max_length=512, truncation=True, padding='max_length', return_tensors='tf')
        targets = tokenizer.encode(summaries[i], max_length=128, truncation=True, padding='max_length', return_tensors='tf')
        predictions = model(inputs)[0]
        rouge.update_state(targets, predictions)
    return rouge.result().numpy()

epochs = 5

for epoch in range(epochs):
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i:i+batch_size]
        batch_summaries = train_summaries[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch_texts, max_length=512, truncation=True, padding='max_length', return_tensors='tf')
        targets = tokenizer.batch_encode_plus(batch_summaries, max_length=128, truncation=True, padding='max_length', return_tensors='tf')
        loss = train_step(inputs['input_ids'], targets['input_ids'])
        print(f"Epoch {epoch + 1}, Batch{int(i/batch_size) + 1} Loss: {loss:.4f}")

    rouge_score = evaluate(test_texts, test_summaries)
    print(f"Epoch {epoch + 1} ROUGE: {rouge_score:.4f}")