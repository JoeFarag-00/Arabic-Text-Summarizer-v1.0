import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from nltk.stem import ISRIStemmer
from pathlib import Path

# Selecting an Arabic Dataset
# The dataset used in this example is "Arabic Books Corpus" 
# from https://www.kaggle.com/mohamedahmedtaha/arabic-books-corpus
# It consists of 1,000 Arabic books of different topics
# The document length varies from book to book, but the average length is around 100,000 words
# No preprocessing steps have been applied to the dataset yet

# Load the data
df = pd.read_csv("Dataset/Arabic_Books.csv", nrows=100)

# Preprocessing
# The specific preprocessing steps required for Arabic text summarization include:
# 1. Removing non-Arabic characters and digits
# 2. Tokenization
# 3. Removing stop words
# 4. Normalizing the text (e.g. removing diacritics, normalizing letters)
# 5. Stemming (optional)

# Load the Arabic stop words
with open(Path("Stopwords/Stopwords_list.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

# Load the pre-trained Arabert model and tokenizer
model = TFAutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

# Define the preprocessing function
stemmer = ISRIStemmer()
def preprocess(text):
    if isinstance(text, str):
        # Remove non-Arabic characters and digits
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
        # Tokenize the text
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        # Remove stop words
        tokens = [token for token in tokens if token not in arabic_stopwords]
        # Normalize the text
        #tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
        #tokens = [token for token in tokens if not token.isdigit()]
        # Stem the text (optional)
        #tokens = [stemmer.stem(token) for token in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''

# Apply the preprocessing function to the data
df['Text'] = df['Text'].apply(preprocess)
df['Summary'] = df['Summary'].apply(preprocess)

# Split the preprocessed dataset intotraining and testing sets
# We will use an 80-20 split for training and testing, respectively
train_size = int(0.8 * len(df))
train_texts = df['Text'][:train_size].tolist()
train_summaries = df['Summary'][:train_size].tolist()
test_texts = df['Text'][train_size:].tolist()
test_summaries = df['Summary'][train_size:].tolist()

# Define the batch size and the number of epochs
batch_size = 8
epochs = 5

# Define the optimizer and loss function for training
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the training step function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)[0]
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Define the evaluation function using the ROUGE metric
import nltk.translate.bleu_score as bleu
from rouge import Rouge 

rouge = Rouge()

def evaluate(texts, summaries):
    rouge_scores = []
    for i in range(len(texts)):
        inputs = tokenizer.encode(texts[i], max_length=512, truncation=True, padding='max_length', return_tensors='tf')
        targets =tokenizer.encode(summaries[i], max_length=128, truncation=True, padding='max_length', return_tensors='tf')
        predictions = model(inputs)[0]
        # Calculate the ROUGE score
        rouge_score = rouge.get_scores(predictions.numpy(), targets.numpy())[0]['rouge-1']['f']
        rouge_scores.append(rouge_score)
    return sum(rouge_scores) / len(rouge_scores)

# Train the model and evaluate on the testing set
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

# Report the evaluation metrics
# We used the ROUGE score as theevaluation metric, which measures the overlap between the generated summary and the reference summary. The ROUGE score ranges from 0 to 1, with higher scores indicating better performance. The average ROUGE score over all the test examples is reported after each epoch.