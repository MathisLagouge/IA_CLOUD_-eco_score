import gensim.downloader as api
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Load pre-trained model and stopwords
model_word2vec = api.load("glove-wiki-gigaword-100")
nltk.download('stopwords')

# Define dictionary for categories
categories = {
    'organic': ['organic', 'health', 'conscious'],
    'climate': ['vegan', 'renewable', 'local', 'compost'],
    'water': ['water', 'conservation', 'preservation'],
    'social': ['social', 'diverse', 'ethical'],
    'governance': ['responsible'],
    'waste': ['sustainable', 'recycled'],
    'adverse': ['greenwash']
}

def preprocess_review(review):
    # Remove punctuation, convert to lowercase, tokenize, and remove stop words
    review = review.translate(str.maketrans('', '', string.punctuation)).lower().split()
    review = [word for word in review if word not in stopwords.words("english") and word != '…']
    return review

def test_similarity(word1, word2):
    similarity = 0
    try:
        similarity =  model_word2vec.similarity(word1, word2)
    except:
        pass
    return similarity

def process_data(path, threshold=0.7):
    processed_reviews = {category: [] for category in categories}
    
    print("Reading data at {0}".format(path))
    df = pd.read_csv(path).dropna().drop_duplicates()
    print(df.head())
    
    print("Number of reviews: {0}".format(len(df)))
    
    for _, row in df.iterrows():
        review = preprocess_review(row['REVIEW'])
        
        for category, words in categories.items():
            if any(test_similarity(review_word, word) > threshold for review_word in review for word in words):
                processed_reviews[category].append(row)
                print("Added review to category {0}".format(category))

    return processed_reviews

def categorize(path, threshold=0.7):
    categorized_reviews = process_data(path, threshold)

    # Create one csv file per category
    for category, reviews in categorized_reviews.items():
        df = pd.DataFrame(reviews, columns=['CITY', 'PLACE', 'REVIEW', 'LINK'])
        df.to_csv(f"Scraping/categories/{category}.csv", index=False)
