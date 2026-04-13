# =========================
# TEXT ANALYTICS LAB 10
# =========================

import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Fix for VS Code plots
import matplotlib
matplotlib.use('TkAgg')

# =========================
# DOWNLOAD NLTK DATA (FIXED)
# =========================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')   # ✅ IMPORTANT FIX
nltk.download('wordnet')

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# =========================
# FIX FILE PATH
# =========================
base_path = os.path.dirname(__file__)

simple_path = os.path.join(base_path, 'simple_dataset.csv')
real_path = os.path.join(base_path, 'restaurant_reviews.csv')

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_review(review):
    review = str(review).lower()
    review = re.sub(r'[^\w\s]', '', review)
    review = re.sub(r'\d+', '', review)

    tokens = word_tokenize(review)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized_tokens)

# =========================
# PART 1: SIMPLE DATASET
# =========================
print("\n===== SIMPLE DATASET =====")

df = pd.read_csv(simple_path)

df['preprocessed_text'] = df['text'].apply(preprocess_review)

print("\nSample Data:")
print(df.head())

# Vectorization
vectorizer = CountVectorizer(max_features=20)
X = vectorizer.fit_transform(df['preprocessed_text'])
term_freq_matrix = X.toarray()
terms = vectorizer.get_feature_names_out()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(term_freq_matrix, xticklabels=terms, cmap='viridis', annot=True)
plt.title("Heatmap - Simple Dataset")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()

# WordCloud (All)
all_text = " ".join(df['preprocessed_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - All Data")
plt.show()

# WordCloud (Positive)
positive_text = " ".join(df[df['category'] == 'Positive']['preprocessed_text'])
positive_wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# WordCloud (Negative)
negative_text = " ".join(df[df['category'] == 'Negative']['preprocessed_text'])
negative_wc = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(positive_wc, interpolation='bilinear')
plt.title("Positive Reviews")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wc, interpolation='bilinear')
plt.title("Negative Reviews")
plt.axis('off')

plt.show()

# Bar Chart
sentiment_counts = df['category'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
plt.title("Sentiment Count - Simple Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# =========================
# PART 2: REAL DATASET
# =========================
print("\n===== REAL DATASET =====")

df2 = pd.read_csv(real_path)

df2['preprocessed_text'] = df2['Review'].apply(preprocess_review)

print("\nSample Data:")
print(df2.head())

# Vectorization
vectorizer = CountVectorizer(max_features=20)
X2 = vectorizer.fit_transform(df2['preprocessed_text'])
term_freq_matrix2 = X2.toarray()
terms2 = vectorizer.get_feature_names_out()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(term_freq_matrix2, xticklabels=terms2, cmap='viridis', annot=True)
plt.title("Heatmap - Real Dataset")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()

# WordCloud (All)
all_text2 = " ".join(df2['preprocessed_text'])
wc_all = WordCloud(width=800, height=400, background_color='white').generate(all_text2)

plt.figure(figsize=(10, 6))
plt.imshow(wc_all, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - All Reviews")
plt.show()

# WordClouds by Sentiment
positive = " ".join(df2[df2['Sentiment'] == 'Positive']['preprocessed_text'])
negative = " ".join(df2[df2['Sentiment'] == 'Negative']['preprocessed_text'])
neutral = " ".join(df2[df2['Sentiment'] == 'Neutral']['preprocessed_text'])

wc_pos = WordCloud(width=800, height=400, background_color='white').generate(positive)
wc_neg = WordCloud(width=800, height=400, background_color='white').generate(negative)
wc_neu = WordCloud(width=800, height=400, background_color='white').generate(neutral)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(wc_pos, interpolation='bilinear')
plt.title("Positive")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wc_neg, interpolation='bilinear')
plt.title("Negative")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wc_neu, interpolation='bilinear')
plt.title("Neutral")
plt.axis('off')

plt.show()

# Sentiment Bar Chart
sent_counts = df2['Sentiment'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(sent_counts.index, sent_counts.values, color=['green', 'red', 'blue'])
plt.title("Sentiment Count - Real Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Category Distribution
cat_counts = df2['Category'].value_counts()

plt.figure(figsize=(6, 4))
cat_counts.plot(kind='bar')
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Topic WordCloud
all_topics = [topic for topics in df2['Topics'].str.split(', ') for topic in topics]
topic_counts = Counter(all_topics)

wc_topics = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_counts)

plt.figure(figsize=(10, 6))
plt.imshow(wc_topics, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Topics")
plt.show()

print("\n===== DONE SUCCESSFULLY =====")