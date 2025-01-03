#import library
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nest_asyncio
nest_asyncio.apply()

import re
import string
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

!pip install xlrd
df = pd.read_excel("/content/data.xlsx")

# Membuat fungsi untuk menghilangkan url
def strip_links(text):
    link_regex = re.compile(r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text
# Membuat fungsi untuk menghilangkan tagar (#), mention (@), spasi ganda
def strip_all_entities(text):
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Membuat fungsi untuk preprocessing
def strip_numeric(text):
    # Mengubah tanda baca menjadi spasi
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus karakter non-alfabet, kecuali spasi
    text = re.sub(r'[^a-z\s]', '', text)
    # Menghapus spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Membuat fungsi untuk mengimplementasikan fungsi preprocessing
def clean_text(text):
    text = strip_links(text)
    text = strip_all_entities(text)
    text = strip_numeric(text)
    return text

df['text'] = df['text'].fillna('').astype(str)
df['text_bersih'] = df['text'].apply(clean_text)

# Mendefinisikan fungsi untuk menghapus karakter angka dari sebuah kolom
def remove_digits_from_column(column):
    def remove_digits(text):
        return re.sub(r'\d+', '', text)
    column_without_digits = column.apply(remove_digits)
    return column_without_digits

# Mendefinisikan fungsi untuk menghapus karakter non-standar
def remove_non_standard_characters(text):
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned_text

# Menghapus karakter angka dari kolom 'text' dan on-standar dari kolom 'text'
df['text'] = remove_digits_from_column(df['text'])
df['text'] = df['text'].apply(remove_non_standard_characters)

# Fungsi untuk membaca data normalisasi dari file CSV dan mengembalikan kamus normalisasi
def load_normalization_data(file_path):
    normalization_dict = {}
    with open(file_path, 'r', encoding='latin-1') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 2:
                normalization_dict[row[0]] = row[1]
    return normalization_dict
# Mendefinisikan fungsi normalize_text
def normalize_text(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

data_file = "/content/norm.csv"
normalization_file = data_file  

df['text_bersih'] = df['text_bersih'].astype(str)
normalization_dict = load_normalization_data(normalization_file)

# Normalisasi teks pada dataframe
df['normalized_text'] = df['text_bersih'].apply(lambda x: normalize_text(x, normalization_dict))

# Menghilangkan kata-kata stopwords dari file stopwords
# Membaca stopwords dari file Excel
stopwords_df = pd.read_excel("/content/stopwords.xlsx", header=None)

# Mengubah DataFrame stopwords menjadi list
words_to_remove = stopwords_df[0].tolist()

# Menghapus kata/frasa dari kolom 'text_bersih' untuk semua baris
for word in words_to_remove:
    df['normalized_text'] = df['normalized_text'].str.replace(r'\b{}\b'.format(word), '', regex=True, case=False)

df.to_excel('bersih.xlsx')

# Memeriksa data duplikat pada DataFrame
duplicate_rows = df[df.duplicated(subset='normalized_text', keep=False)]

# Menghitung jumlah baris yang duplikat
duplicate_count = len(duplicate_rows)

# Menampilkan hasil
print("Baris yang duplikat:")
print(duplicate_rows)
print("\nJumlah baris yang duplikat:", duplicate_count)

# Menghapus baris yang duplikat dan menyisakan satu saja
df= df.drop_duplicates(subset='normalized_text', keep='first')

# Menampilkan hasil
print("DataFrame setelah menghapus duplikat:")
print(df)

any_missing = df.isnull().values.any()

# Menampilkan hasil
if any_missing:
    print("Ada nilai kosong dalam DataFrame.")
else:
    print("Tidak ada nilai kosong dalam DataFrame.")

# Mengecek dan menampilkan baris dengan nilai kosong
rows_with_missing_values = df[df.isnull().any(axis=1)]
# Menampilkan jumlah baris dari subset DataFrame
num_rows_with_missing_values = rows_with_missing_values.shape[0]

# Menampilkan hasil
print("Baris dengan nilai kosong:")
print(rows_with_missing_values)
print("\nJumlah baris dengan nilai kosong:", num_rows_with_missing_values)

# Menghapus baris dengan nilai kosong dari DataFrame asli
df.dropna(inplace=True)

# Menampilkan hasil
print("DataFrame asli setelah menghapus baris dengan nilai kosong:")

# Memeriksa pemberian label kandidat capres-cawapres
label_counts = df[['label','keyword']].value_counts()
print(label_counts)

# Mengoreksi kesalahan pada label pasangan
kondisi = [
    ('anies', 'anies'),
    ('imin', 'anies'),
    ('prabowo', 'prabowo'),
    ('gibran', 'prabowo'),
    ('ganjar', 'ganjar'),
    ('mahfud', 'ganjar'),
    ('imin', 'ganjar'),
    ('anies', 'prabowo')
]

# Menginisialisasi jumlah baris yang memenuhi semua kondisi
hasil_kondisi = {}

# Melakukan iterasi melalui setiap pasangan nilai dan menghitung jumlah baris yang memenuhi kondisi
for pasangan in kondisi:
    hasil_kondisi[pasangan] = df[(df['keyword'] == pasangan[0]) & (df['label'] == pasangan[1])].shape[0]

# Menampilkan hasil untuk setiap kondisi
for pasangan, jumlah_baris in hasil_kondisi.items():
    print("Jumlah baris yang memenuhi kondisi {}: {}".format(pasangan, jumlah_baris))

# Memeriksa apakah ada nilai kosong di kolom label
ada_kosong = df['label'].isnull().any()

if ada_kosong:
    print("Ada nilai kosong di kolom label.")
else:
    print("Tidak ada nilai kosong di kolom label.")

# Mengoreksi pasangan label
def ganti_label(row):
    if row['keyword'] in ['anies', 'imin']:
        return 'anies'
    elif row['keyword'] in ['ganjar', 'mahfud']:
        return 'ganjar'
    elif row['keyword'] in ['prabowo', 'gibran']:
        return 'prabowo'
    else:
        return None

# Mengganti nilai label berdasarkan pasangan yang sesuai
df['label'] = df.apply(ganti_label, axis=1)

# Menampilkan hasil untuk setiap kondisi
hasil_kondisi = {}

# Iterasi melalui setiap pasangan nilai dan menghitung jumlah baris yang memenuhi kondisi
for pasangan in kondisi:
    hasil_kondisi[pasangan] = df[(df['keyword'] == pasangan[0]) & (df['label'] == pasangan[1])].shape[0]

for pasangan, jumlah_baris in hasil_kondisi.items():
    print("Jumlah baris yang memenuhi kondisi {}: {}".format(pasangan, jumlah_baris))

df.to_excel("bersih_Akhir.xlsx")

# Memeriksa dan menghitung jumlah data duplikat 
duplicate_rows = df[df.duplicated(subset='text', keep=False)]
duplicate_count = len(duplicate_rows)

# Menampilkan hasil
print("Baris yang duplikat:")
print(duplicate_rows)
print("\nJumlah baris yang duplikat:", duplicate_count)

# Menghapus baris yang duplikat dan menyisakan satu saja
df= df.drop_duplicates(subset='text', keep='first')

# Menampilkan hasil
print("DataFrame setelah menghapus duplikat:")

# Memeriksa missing value
any_missing = df.isnull().values.any()

# Menampilkan hasil jika ada
if any_missing:
    print("Ada nilai kosong dalam DataFrame.")
else:
    print("Tidak ada nilai kosong dalam DataFrame.")

# Memeriksa dan menampilkan baris dengan nilai kosong
rows_with_missing_values = df[df.isnull().any(axis=1)]
num_rows_with_missing_values = rows_with_missing_values.shape[0]

# Menampilkan hasil jika ada
print("Baris dengan nilai kosong:")
print(rows_with_missing_values)
print("\nJumlah baris dengan nilai kosong:", num_rows_with_missing_values)

# Stemming
!pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Mendefinisikan fungsi untuk melakukan stemming pada teks
def stem_text(text):
    return stemmer.stem(text)

# Menerapkan fungsi stemming pada kolom 'normalized_text'
df['stemmed_text'] = df['text'].apply(stem_text)

# Menampilkan DataFrame hasil stemming
print(df)

# Memeriksa data duplikat setelah proses stemming
duplicate_rows = df[df.duplicated(subset='stemmed_text', keep=False)]

# Menghitung jumlah baris yang duplikat
duplicate_count = len(duplicate_rows)

# Menampilkan hasil
print("Baris yang duplikat:")
print(duplicate_rows)
print("\nJumlah baris yang duplikat:", duplicate_count)

# Menghapus baris yang duplikat dan menyisakan satu saja
df= df.drop_duplicates(subset='stemmed_text', keep='first')

# Memeriksa missing value setelah proses stemming
any_missing = df.isnull().values.any()

# Menampilkan hasil
if any_missing:
    print("Ada nilai kosong dalam DataFrame.")
else:
    print("Tidak ada nilai kosong dalam DataFrame.")

# Memeriksa dan menampilkan baris dengan nilai kosong
rows_with_missing_values = df[df.isnull().any(axis=1)]
# Menampilkan jumlah baris dari subset DataFrame
num_rows_with_missing_values = rows_with_missing_values.shape[0]

# Menampilkan hasil
print("Baris dengan nilai kosong:")
print(rows_with_missing_values)
print("\nJumlah baris dengan nilai kosong:", num_rows_with_missing_values)

# Menyimpan DataFrame setelah proses stemming
df.to_excel("norm_stem.xlsx")

# Menghapus Stopwords dari Sastrawi
!pip install Sastrawi

# Mengimpor pustaka yang diperlukan
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Membuat objek stemmer dari StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Memuat stop words dalam bahasa Inggris dan bahasa Indonesia
stop_words_en = set(stopwords.words('english'))
stop_words_id = set(stopwords.words('indonesian'))

# Memuat stop words Bahasa Indonesia dari library Sastrawi
stopword_factory = StopWordRemoverFactory()
stop_sastrawi_id = stopword_factory.get_stop_words()

# Menggabungkan stopwords dari kedua sumber bahasa Indonesia
stop_words_id = stop_words_id.union(stop_sastrawi_id)

# Mendefinisikan fungsi untuk menghapus stopwords dalam bahasa Indonesia dan bahasa Inggris
def remove_stopwords_id(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words_id]
    return ' '.join(filtered_tokens)

def remove_stopwords_en(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words_en]
    return ' '.join(filtered_tokens)

# Menerapkan fungsi stemming dan menghapus stopwords
df['stemmed_text'] = df['text'].apply(stemmer.stem)
df['stemmed_text'] = df['stemmed_text'].apply(remove_stopwords_id)

# Menampilkan DataFrame hasil akhir
print(df)

# Menyimpan DataFrame setelah proses menghilangkan stopwords
df.to_excel("norm_stem_stop.xlsx", index=False)