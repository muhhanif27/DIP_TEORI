import googleapiclient.discovery
import pymongo
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER untuk analisis sentimen
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Konfigurasi API YouTube
API_KEY = "_api_key"  # Ganti dengan API Key kamu
VIDEO_ID = "NYH6Oa4PXlY"  # Ganti dengan ID video YouTube yang diinginkan

# Koneksi ke MongoDB (ganti connection string jika pakai MongoDB Atlas)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["youtube"]
collection = db["comments"]

# **STEP 1: EXTRACT - Mengambil Semua Komentar dari YouTube**
def extract_comments(video_id, max_comments=18000):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None
    total_extracted = 0

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Maksimum per request
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            # Cek apakah komentar sudah ada di database (hindari duplikasi)
            if not collection.find_one({"comment": comment_text}):
                comments.append({"comment": comment_text})
                total_extracted += 1

        # Update token untuk halaman berikutnya
        next_page_token = response.get("nextPageToken")

        print(f"Total comments extracted so far: {total_extracted}")  # Progress log

        # Berhenti jika tidak ada halaman berikutnya atau sudah mencapai batas
        if not next_page_token or total_extracted >= max_comments:
            break

    return comments

# **STEP 2: LOAD - Simpan Komentar ke MongoDB**
def load_comments_to_mongodb(comments):
    if comments:
        collection.insert_many(comments)
        print(f"{len(comments)} new comments saved to MongoDB!")
    else:
        print("No new comments to save.")

# **STEP 3: TRANSFORM - Analisis Sentimen dan Update MongoDB**
def transform_comments():
    comments = collection.find({"sentiment": {"$exists": False}})  # Hanya komentar baru

    for item in comments:
        sentiment_score = sia.polarity_scores(item["comment"])
        if sentiment_score['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        collection.update_one(
            {"_id": item["_id"]},
            {"$set": {"sentiment": sentiment, "score": sentiment_score}}
        )

    print("Sentiment analysis completed and updated in MongoDB!")

# **STEP 4: VISUALISASI - Pie Chart Sentimen**
def visualize_sentiment():
    sentiment_counts = {
        "Positive": collection.count_documents({"sentiment": "Positive"}),
        "Neutral": collection.count_documents({"sentiment": "Neutral"}),
        "Negative": collection.count_documents({"sentiment": "Negative"})
    }

    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['green', 'gray', 'red']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Sentiment Analysis of YouTube Comments")
    plt.show()

# **Jalankan Proses ELT dan Visualisasi**
comments = extract_comments(VIDEO_ID, max_comments=18000)  # Extract
load_comments_to_mongodb(comments)  # Load
transform_comments()  # Transform
visualize_sentiment()  # Visualisasi
