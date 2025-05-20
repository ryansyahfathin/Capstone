import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import gdown
import os


# Title dan Deskripsi
st.title("📊 Analisis Dataset Sosial Media")
st.write("Mengunduh dataset dari Google Drive dan melakukan preprocessing awal.")

# Unduh file dari Google Drive
file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'dataset_social_media.xlsx'

# Unduh hanya jika belum ada
if not os.path.exists(output):
    with st.spinner('Mengunduh dataset dari Google Drive...'):
        gdown.download(url, output, quiet=False)

# Load dataset
df = pd.read_excel(output, sheet_name='Working File')

# Tampilkan preview data
st.subheader("🔍 Preview Data")
st.dataframe(df.head())

# Preprocessing
st.subheader("🔧 Preprocessing Awal")
df['Platform'] = df['Platform'].astype(str).str.strip().str.title()
df['Post Type'] = df['Post Type'].astype(str).str.strip().str.title()
df['Audience Gender'] = df['Audience Gender'].astype(str).str.strip().str.title()
df['Age Group'] = df['Age Group'].astype(str).str.strip().str.title()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.title()
df['Time Periods'] = df['Time Periods'].astype(str).str.strip().str.title()
df['Weekday Type'] = df['Weekday Type'].astype(str).str.strip().str.title()


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import os

st.subheader("🧹 Pembersihan Kolom Tidak Relevan")

# Drop kolom yang tidak relevan
drop_cols = [
    'Post ID', 'Date', 'Time',
    'Audience Location', 'Audience Continent', 'Audience Interests',
    'Campaign ID', 'Influencer ID',
    'Weekday Type'
]
df.drop(columns=drop_cols, inplace=True)

st.success("Kolom tidak relevan telah dibuang.")

# Inisialisasi Analisis Sentimen
st.subheader("💬 Inisialisasi Sentimen VADER dan IndoBERT")

# Download VADER Lexicon
with st.spinner("Mengunduh lexicon VADER..."):
    nltk.download('vader_lexicon')

# Inisialisasi VADER dan IndoBERT
try:
    vader_analyzer = SentimentIntensityAnalyzer()
    indo_sentiment = pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p1")
    st.success("Model VADER dan IndoBERT siap digunakan ✅")
except Exception as e:
    st.error(f"Gagal inisialisasi model: {e}")

# --- ANALISIS SENTIMEN VADER ---
st.subheader("💬 Analisis Sentimen Caption")
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(caption):
    score = vader_analyzer.polarity_scores(caption)
    sentiment = "Positive" if score['compound'] >= 0.05 else "Negative" if score['compound'] <= -0.05 else "Neutral"
    return sentiment

# Tambahkan kolom waktu dari timestamp
if 'Post Timestamp' in df.columns:
    df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], errors='coerce')
    df['Post Hour'] = df['Post Timestamp'].dt.hour
    df['Post Day Name'] = df['Post Timestamp'].dt.day_name()
else:
    st.warning("Kolom 'Post Timestamp' tidak ditemukan.")

# === FUNGSI Rekomendasi dan Prediksi ===
def hybrid_recommendation_pipeline_super_adaptive(post_type, audience_gender, age_group, sentiment=None, platform_input=None):
    warning_text = ""
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    if sentiment:
        filtered_sent = filtered[filtered['Sentiment'] == sentiment]
    else:
        filtered_sent = filtered

    if not platform_input or platform_input.lower() == 'all':
        group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
        filtered_sent_platform = filtered_sent
    else:
        filtered_sent_platform = filtered_sent[filtered_sent['Platform'] == platform_input.title()]
        group_cols = ['Time Periods', 'Post Day Name', 'Post Hour']

    main_reco = (
        filtered_sent_platform.groupby(group_cols)
        .agg({'Engagement Rate': 'mean'})
        .sort_values('Engagement Rate', ascending=False)
        .reset_index()
    )

    if main_reco.empty and platform_input and platform_input.lower() != 'all':
        warning_text += "Data terlalu sempit dengan filter platform.\n"
        group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
        main_reco = (
            filtered_sent.groupby(group_cols)
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )

    if main_reco.empty and sentiment:
        warning_text += "Data terlalu sempit dengan filter sentiment.\n"
        filtered_no_sent = df[
            (df['Post Type'] == post_type) &
            (df['Audience Gender'] == audience_gender) &
            (df['Age Group'] == age_group)
        ]
        if platform_input and platform_input.lower() != 'all':
            filtered_no_sent_platform = filtered_no_sent[filtered_no_sent['Platform'] == platform_input.title()]
            group_cols = ['Time Periods', 'Post Day Name', 'Post Hour']
            main_reco = (
                filtered_no_sent_platform.groupby(group_cols)
                .agg({'Engagement Rate': 'mean'})
                .sort_values('Engagement Rate', ascending=False)
                .reset_index()
            )
        else:
            group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']
            main_reco = (
                filtered_no_sent.groupby(group_cols)
                .agg({'Engagement Rate': 'mean'})
                .sort_values('Engagement Rate', ascending=False)
                .reset_index()
            )

    if main_reco.empty:
        warning_text += "Data sangat sempit, memberikan rekomendasi umum untuk post type saja.\n"
        main_reco = (
            df[df['Post Type'] == post_type]
            .groupby(['Platform', 'Time Periods', 'Post Day Name', 'Post Hour'])
            .agg({'Engagement Rate': 'mean'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )

    return main_reco.head(5), warning_text

def strategy_recommendation(post_type, audience_gender, age_group):
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    strategy = (
        filtered.groupby('Sentiment')
        .agg({'Engagement Rate': 'mean', 'Post Content': 'count'})
        .rename(columns={'Post Content': 'Jumlah Post'})
        .sort_values('Engagement Rate', ascending=False)
        .reset_index()
    )
    if strategy.empty:
        strategy = (
            df[df['Post Type'] == post_type]
            .groupby('Sentiment')
            .agg({'Engagement Rate': 'mean', 'Post Content': 'count'})
            .rename(columns={'Post Content': 'Jumlah Post'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
    return strategy

def alternative_platform_suggestion(post_type, audience_gender, age_group, platform_input):
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    if platform_input and platform_input.lower() != 'all':
        alt_platform_stats = (
            filtered.groupby('Platform')
            .agg({'Engagement Rate': 'mean', 'Post Content': 'count'})
            .rename(columns={'Post Content': 'Jumlah Post'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
        alt_platform_stats = alt_platform_stats[alt_platform_stats['Platform'] != platform_input.title()]
    else:
        alt_platform_stats = (
            filtered.groupby('Platform')
            .agg({'Engagement Rate': 'mean', 'Post Content': 'count'})
            .rename(columns={'Post Content': 'Jumlah Post'})
            .sort_values('Engagement Rate', ascending=False)
            .reset_index()
        )
    return alt_platform_stats.head(3)

def engagement_rate_prediction():
    features = df[['Likes', 'Comments', 'Shares', 'Impressions', 'Reach']]
    target = df['Engagement Rate']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, rmse

st.header("🧠 Sistem Rekomendasi Konten Sosial Media")

# --- Form Input User ---
with st.form("recommendation_form"):
    caption_input = st.text_input("Masukkan Caption Anda")
    post_type_input = st.selectbox("Pilih Jenis Post (Post Type):", ["Video", "Image", "Link"])
    audience_gender_input = st.selectbox("Pilih Gender Audiens:", ["Male", "Female", "Other"])
    age_group_input = st.selectbox("Pilih Kelompok Umur:", ["Senior Adults", "Mature Adults", "Adolescent Adults"])
    platform_input = st.selectbox("Pilih Platform (atau All):", ["All", "Instagram", "Facebook", "Twitter", "LinkedIn"])

    submit_button = st.form_submit_button(label="🔍 Prediksi & Rekomendasikan")

# Jika tombol ditekan
if submit_button:
    sentiment_detected = analyze_sentiment(caption_input)

    st.success(f"✅ Prediksi Sentimen Caption Anda: {sentiment_detected}")

    # Pipeline Rekomendasi
    reco_pipeline, warning_pipeline = hybrid_recommendation_pipeline_super_adaptive(
        post_type_input,
        audience_gender_input,
        age_group_input,
        sentiment_detected,
        platform_input
    )

    if not reco_pipeline.empty:
        best_reco = reco_pipeline.iloc[0]
        if 'Platform' in reco_pipeline.columns:
            reco_text = f"Post pada pukul {int(best_reco['Post Hour']):02d}:00 WIB di hari {best_reco['Post Day Name']} melalui platform {best_reco['Platform']} untuk engagement maksimal."
        else:
            reco_text = f"Post pada pukul {int(best_reco['Post Hour']):02d}:00 WIB di hari {best_reco['Post Day Name']} untuk engagement maksimal."
    else:
        reco_text = "Tidak ada rekomendasi yang cukup relevan berdasarkan input Anda."

    st.markdown(f"### 🎯 Rekomendasi Waktu Posting:\n> {reco_text}")

    # Strategi Caption
    strategy_reco = strategy_recommendation(post_type_input, audience_gender_input, age_group_input)
    if not strategy_reco.empty:
        best_strategy = strategy_reco.iloc[0]
        strategy_text = f"Gunakan konten {post_type_input.lower()} dengan sentimen {best_strategy['Sentiment'].lower()} untuk {age_group_input.lower()}."
    else:
        strategy_text = "Tidak ditemukan strategi caption yang relevan."
    
    st.markdown(f"### 💡 Strategi Konten:\n> {strategy_text}")

    # Alternatif Platform
    alt_platform_reco = alternative_platform_suggestion(
        post_type_input,
        audience_gender_input,
        age_group_input,
        platform_input
    )

    if not alt_platform_reco.empty:
        if platform_input.lower() != 'all':
            alt_platform_text = f"Platform alternatif yang dapat Anda pertimbangkan: {alt_platform_reco.iloc[0]['Platform']}."
        else:
            alt_platform_text = "Platform alternatif yang dapat Anda pertimbangkan: " + ", ".join(alt_platform_reco['Platform'].tolist())
    else:
        alt_platform_text = "Tidak ada platform alternatif yang disarankan."

    st.markdown(f"### 🔄 Saran Platform Alternatif:\n> {alt_platform_text}")

    if warning_pipeline:
        st.warning(f"⚠️ Catatan:\n{warning_pipeline}")
