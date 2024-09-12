from flask import Flask, jsonify, request
from flask_cors import CORS
from twikit import Client
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline, AutoTokenizer
import socket
import asyncio
import nest_asyncio
socket.setdefaulttimeout(100)

nest_asyncio.apply()


# Initialize Flask app
app = Flask(__name__)
CORS(app) 


#twitter credentials
USERNAME = ''
EMAIL = ''
PASSWORD = ''


client = Client('en-ID')

async def login_client():
    # client.load_cookies('cookies.json')
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD
    )

asyncio.run(login_client())

stop_words_indo = set(stopwords.words('indonesian'))

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Define Indonesian punctuation marks to remove
    indo_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    
    # Remove punctuation
    text = re.sub(r'[{}]'.format(re.escape(indo_punctuation)), '', text)
    
    # Tokenize text using NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize (stem) tokens using Sastrawi
    lemmatized_tokens = []
    for token in tokens:
        if token not in stop_words_indo:
            lemma = stemmer.stem(token)
            lemmatized_tokens.append(lemma)
    
    # Join lemmatized tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

# Choose a model specifically trained for Bahasa Indonesia
model_name = "mdhugol/indonesia-bert-sentiment-classification"

# Load tokenizer and pre-trained sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

async def get_tweet_next(tweets):
    tweets = await tweets.next()
    return tweets

def get_sentiment (tweets, count, positive_count_args, negative_count_args) :
    response_data = []
    positive_count = positive_count_args
    negative_count = negative_count_args
    i = 0
    countz = count
    
    for tweet in tweets:
        i = i +1
        countz = countz + 1
        text = tweet.text
        preprocessed_text = preprocess_text(text)
        result = classifier(preprocessed_text)
        label = "Positive" if result[0]['label'] == "LABEL_0" else "Neutral" if result[0]['label'] == "LABEL_1" else "Negative"
        
        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
        
        response_data.append({
            'tweet': text,
            'preprocessed_tweet': preprocessed_text,
            'sentiment': label,
            'score': result[0]['score']
        })
        if i == len(tweets) - 1 and countz < 100:
            next = asyncio.run(get_tweet_next(tweets=tweets)) 
            positive_count,negative_count, response_next, countz  = get_sentiment(next, countz, positive_count, negative_count)
            response_data = response_data + response_next
    return positive_count, negative_count,response_data, countz

async def get_tweet(query,count):
    tweets = await client.search_tweet(query, 'Latest', count)
    return tweets

@app.route('/analyze_tweets', methods=['GET'])
def analyze_tweets():
    query = request.args.get('query')
    count = int(request.args.get('count', 20))
    tweets = asyncio.run(get_tweet(query=query, count=count))
    
    
    positive_count,negative_count,response_data, countz = get_sentiment(tweets, 0, 0,0)
    
    positive_percentage = (positive_count / countz) * 100 if countz > 0 else 0
    
    response = {
        'tweets': response_data,
        'positive_percentage': positive_percentage,
        'positive_counnt': positive_count,
        'negative_count' : negative_count,
        'total': countz,
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
