import os
from dotenv import load_dotenv
from flask import Flask ,render_template,request, url_for, redirect
from convex import ConvexClient
from keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
nltk.download('punkt')

load_dotenv(".env.local")
load_dotenv()

client = ConvexClient(os.getenv("CONVEX_URL"))

app = Flask(__name__)

global prompt
global output
prompt = "error"
output="undetermined"

@app.route('/predict',methods= ['POST', 'GET'])
def predict():
    # Your function logic here
    
    if request.method == "POST":
        global prompt
        prompt = request.form["address"]
    
    model = load_model("saved_model_final4.keras")
    model.summary()

    # New texts
    new_texts = [prompt]
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
    #print(predictions)
    # Output predictions
    sentiment_labels = ["Negative", "Positive"]
    #for i, text in enumerate(new_texts):
    predicted_class = np.argmax(predictions)
        #print(f"Text: {text}\nPredicted Sentiment Class: {sentiment_labels[predicted_class]}\n")
    global output
    if (predicted_class==0):
        output="negative"
    else:
        output="positive"

    client.mutation("history:send", dict(prompt=prompt))
    return redirect(url_for("page_3", output_data = output))


@app.route('/feedback',methods= ['POST', 'GET'])
def feedback():
    model_emo = output
    if request.method == "POST":
        user_emo = request.form["feedback"]
        if user_emo=='Yes':
            user_emo=model_emo
        else:
            if model_emo=="positive":
                user_emo="negative"
            elif model_emo=="negative":
                user_emo="positive"
            
       
        client.mutation("feedback:send", dict(prompt=prompt, model_emo=model_emo, user_emo=user_emo))
        # return redirect(url_for("page_1"))
        return render_template('page_1.html')

@app.route('/')
def page_1():
    return render_template('page_1.html')

@app.route('/page_1')
def page_11():
    return render_template('page_1.html')


@app.route('/page_2')  # Define the route with the correct endpoint
def page_2():
    return render_template('page_2.html')


@app.route('/page_3/<output_data>' , methods=['GET', 'POST']) # Define the route with the correct endpoint
def page_3(output_data):
    return render_template('page_3.html', data = output_data)


@app.route('/about')  # Define the route with the correct endpoint
def about():
    return render_template('about.html')


@app.route('/how_to_play')  # Define the route with the correct endpoint
def how_to_play():
    return render_template('how_to_play.html')


@app.route('/contact_us')  # Define the route with the correct endpoint
def contact_us():
    return render_template('contact_us.html')

if __name__ == '__main__':
    app.run(debug=True)
