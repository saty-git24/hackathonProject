import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('output2.csv')

print(df.columns)

df.fillna({'stemmed_content': ''}, inplace=True)

print(df.isnull().sum())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index

df['tokenized_text'] = tokenizer.texts_to_sequences(df['text'])
print(df['tokenized_text'])

padded_sequences = pad_sequences(df['tokenized_text'], maxlen=100, padding='post', truncating='post')

# Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)
num_classes = 6  # 0 to 5 sentiments

one_hot_labels = to_categorical(df['label'], num_classes=num_classes)

model = Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

epochs = 1  # You can adjust this based on your needs
batch_size = 128
model.fit(padded_sequences, one_hot_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

# To save the model
model.save('saved_model_final.keras')
