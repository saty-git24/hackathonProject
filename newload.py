import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import re
import nltk
nltk.download('punkt')

# Load the pre-trained model
model = load_model("saved_model_final4.keras")
model.summary()

# New texts
new_texts = ["ive been feeling so emotional and nostalgic over the last few weeks",
             "i feel heartbroken hurt attacked displaced and separated",
             "i feel thrilled", "wow i can't believe you are here",
             "i feel awful for a past couple of weeks",
             "i have been successful in providing some peace of mind i feel content",
             "i feel free to express myself without inhibition",
             "have you gone mad",
             "i mean i imagine ive put on some muscle as ive been jogging more quickly but wow i never imagined id get to that weight and not feel disgusted with myself at first",
             "i feel like she s extremely talented"]

# Convert to DataFrame
df = pd.DataFrame(new_texts, columns=["Text"])

# Define text preprocessing functions
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize words
    words = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Perform stemming
    porter = nltk.PorterStemmer()
    words = [porter.stem(word) for word in words]
    # Perform lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply text preprocessing
df['Text'] = df['Text'].apply(clean_text)

# Tokenization and padding
max_features = 1000  # Adjust this according to your data
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['Text'].values)
X = tokenizer.texts_to_sequences(df['Text'].values)
maxlen = 34
X = pad_sequences(X, maxlen=maxlen)

# Now X contains the preprocessed and tokenized text data
predictions = model.predict(X)
print(predictions)
# Output predictions
sentiment_labels = ["Negative", "Positive"]
for i, text in enumerate(new_texts):
    predicted_class = np.argmax(predictions[i])
    print(f"Text: {text}\nPredicted Sentiment Class: {sentiment_labels[predicted_class]}\n")

































'''''
# Example usage for prediction
new_texts = ["ive been feeling so emotional and nostalgic over the last few weeks", "i feel heartbroken hurt attacked displaced and separated"
             , "i feel thrilled", "wow i can't believe you are here", "i feel awful for a past couple of weeks",
             "i have been successful in providing some peace of mind i feel content","i feel free to express myself without inhibition",
             "have you gone mad","i mean i imagine ive put on some muscle as ive been jogging more quickly but wow i never imagined id get to that weight and not feel disgusted with myself at first","i feel like she s extremely talented"]
# Convert to DataFrame
df = pd.DataFrame(new_texts, columns=["Text"])


# Stemming of df["Text"]
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import re
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('^a-zA-z',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return(stemmed_content)


df['stemmed_content'] = df['Text'].apply(stemming)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["Text"])

df['Text'] = tokenizer.texts_to_sequences(df['Text'])
print(df['Text'])

padded_sequences = pad_sequences(df['Text'], maxlen=100, padding='post', truncating='post')

predictions = model.predict(padded_sequences)
print(predictions)
# Output predictions
sentiment_labels = ["Negative", "Positive"]
for i, text in enumerate(new_texts):
    predicted_class = np.argmax(predictions[i])
    print(f"Text: {text}\nPredicted Sentiment Class: {sentiment_labels[predicted_class]}\n")

for i, text in enumerate(new_texts):
    predicted_class = np.argmax(predictions[i])
    print(f"Text: {text}\nPredicted Sentiment Class: {predicted_class}\n")
'''''
