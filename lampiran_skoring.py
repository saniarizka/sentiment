import numpy as np
import pandas as pd

!pip install xlrd

data = pd.read_excel("/content/data_trans.xlsx")

data.info()

data['text_en'] = data['text_en'].fillna('').astype(str)

# Memproses teks untuk menghapus tanda baca dan karakter non-alfanumerik
def remove_non_alphanumeric(text):
    text_cleaned = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text_cleaned

# Menerapkan preprocessing pada kolom 'Text' dan case folding
data['text_en'] = data['text_en'].apply(remove_non_alphanumeric)
data['text_en'] = data['text_en'].str.lower()

"""Labelling score dengan VADER pada text berbahasa Inggris"""

!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Inisialisasi objek SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Terapkan analisis sentimen pada setiap teks dalam DataFrame
data['sentiment_scores'] = data['text_en'].apply(lambda x: analyzer.polarity_scores(x))

# Ekstrak skor sentimen positif dan negatif ke dalam kolom baru
data['positive_score'] = data['sentiment_scores'].apply(lambda x: x['pos'])
data['negative_score'] = data['sentiment_scores'].apply(lambda x: x['neg'])
data['neutral_score'] = data['sentiment_scores'].apply(lambda x: x['neu'])
data['compound_score'] = data['sentiment_scores'].apply(lambda x: x['compound'])

data

#Membuat fungsi untuk pemberian label skor negatif, netral, dan positif
def encode_sentiment(score):
    if score['compound'] > 0:
        return 'positive'
    elif score['compound'] < 0 :
        return 'negative'
    else:
        return 'neutral'

# Menerapkan pengelompokan sentimen pada data
data['sentiment_encoded'] = data['sentiment_scores'].apply(encode_sentiment)

# Menghitung jumlah sentimen
sentiment_counts = data['sentiment_encoded'].value_counts()

# Menghitung jumlah tiap label sentimen berdasarkan label tertentu
sentiment_counts_by_label = data.groupby('label')['sentiment_encoded'].value_counts()

# Menampilkan hasil perhitungan
print(sentiment_counts)
print(sentiment_counts_by_label)

data.replace('', np.nan, inplace=True)

# Memeriksa nilai kosong pada data
baris_kosong = data.isna().any(axis=1)
print("Baris yang memiliki nilai kosong di setidaknya satu kolom:")
print(data[baris_kosong])

# Jumlah baris yang memiliki nilai kosong di semua kolom
jumlah_baris_kosong = baris_kosong.sum()
print(f"Jumlah baris dengan nilai kosong di semua kolom: {jumlah_baris_kosong}")

# Hapus baris yang memiliki nilai NaN di kolom teks
data_clean = data.dropna(subset=['text_en'])

# Memeriksa duplikat berdasarkan kolom 'col1'
duplikat = data_clean.duplicated(subset=['text_en'])
print("Duplikat berdasarkan kolom 'text_en':")
data_clean[duplikat]

# Menghapus duplikat berdasarkan kolom data teks dan menyimpan baris terakhir
data_clean = data_clean.drop_duplicates(subset=['text_en'], keep='last')

# Memeriksa hasil akhir tahap preprocessing data
data

# Menyimpan data ke file
data_clean.to_excel("trans_nostem_nostop.xlsx")