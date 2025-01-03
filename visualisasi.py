import numpy as np
import pandas as pd

df = pd.read_excel("/content/trans_nostem_nostop.xlsx")

# Menghitung jumlah sentimen dalam data
sentiment_counts = df['sentiment_encoded'].value_counts()

# Menampilkan hasil perhitungan
print(sentiment_counts)

# Mengitung jumlah tiap label sentimen berdasarkan keywords (kandidat)
sentiment_counts_by_label = df.groupby('label')['sentiment_encoded'].value_counts()

# Menampilkan hasil perhitungan
print(sentiment_counts_by_label)

"""Visualisasi Word Clouds"""

# Mengambil kolom data tertentu yang diperlukan dalam visualisasi
df = df[["text_in", "text_en","keyword", "label", "sentiment_encoded"]]
col_names = ['text_in', 'text_en', 'keyword', 'label', 'category']
df.columns=col_names

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.util import ngrams
from collections import defaultdict, Counter

import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Memastikan data bertipe string
df['text_en'] = df['text_en'].astype(str)

#Memuat stop words Inggris dari library NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
# Membuat fungsi untuk menghapus stopwords dari dataset

def remove_stopwords_en(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words_en]
    return ' '.join(filtered_tokens)

# Menerapkan fungsi untuk menghapus stopwords yang telah digabungkan
df['text_en'] = df['text_en'].apply(remove_stopwords_en)

# Visualisasi word clouds data teks secara keseluruhan
# Membuat corpus untuk kasus all keyword
corpus = ' '.join(df['text_in']).split()
corpus_pos = ' '.join(df[df['category'] == 'positive']['text_en']).split()
corpus_neg = ' '.join(df[df['category'] == 'negative']['text_en']).split()
corpus_net = ' '.join(df[df['category'] == 'neutral']['text_en']).split()

# Menghitung jumlah kata pada setiap corpus
corpus_count = Counter(corpus).most_common()
corpus_pos_count = Counter(corpus_pos).most_common()
corpus_neg_count = Counter(corpus_neg).most_common()
corpus_net_count = Counter(corpus_net).most_common()

# Menampilkan informasi
print("Ini merupakan visualisasi untuk data secara keseluruhan tanpa membedakan label")
print("Jumlah kata unik pada data keseluruhan:", len(corpus_count))
print("Jumlah kata unik pada sentimen positif:", len(corpus_pos_count))
print("Jumlah kata unik pada sentimen negatif:", len(corpus_neg_count))
print("Jumlah kata unik pada sentimen netral:", len(corpus_net_count))

# Menampilkan word clouds
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[15, 7])

wordcloud1 = WordCloud(width=800, height=800, background_color='white', colormap="RdPu",collocations=False,
                      relative_scaling = 0.2, min_font_size=10).generate(' '.join(corpus_pos))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Word Cloud Ulasan Positif')

wordcloud2 = WordCloud(width=800, height=800, background_color='white', colormap="Reds", collocations=False,
                      relative_scaling = 0.2, min_font_size=10).generate(' '.join(corpus_neg))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Word Cloud Ulasan Negatif')

wordcloud3 = WordCloud(width=800, height=800, background_color='white', colormap="Blues", collocations=False,
                      relative_scaling = 0.2, min_font_size=10).generate(' '.join(corpus_net))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Word Cloud Ulasan Netral')
plt.show()

# Menyalin dataframe df
df1=df

stopwords_anies = ['anies', 'cak', 'imin', 'people', 'candidate', 'number', 'presidential', 'indonesia', 'imins',
                   'muhaimin', 'iskandar', 'baswedan', 'prabowo', 'im', 'gibran', 'ganjar', 'reward', 'mahfud',
                   'gangar', 'gangars', 'gibrans', 'md', 'mahfuds', 'mds', 'soebianto', 'subianto', 'rakabuming',
                   'professor', 'bro', 'really', 'milu', 'brother', 'mas', 'peoples', 'pranowo', 'indonesian',
                   'candidates', 'rewards', 'prof', 'already', 'prabowos', 'vice', 'debate', 'president',
                   'support', 'election'
                   ]

stop_words_en.update(stopwords_anies)
# Menerapkan remove stopwords
df1['text_en'] = df1['text_en'].apply(remove_stopwords_en)

# Visualisasi word clouds untuk pasangan calon Anies-Cak Imin
# Memfilter paslon atau keyword
df_anies = df1[df1['label'] == 'anies']

# Membuat corpus untuk kasus all keyword
corpus1 = ' '.join(df_anies['text_en']).split()
corpus_pos1 = ' '.join(df_anies[df_anies['category'] == 'positive']['text_en']).split()
corpus_neg1 = ' '.join(df_anies[df_anies['category'] == 'negative']['text_en']).split()
corpus_net1 = ' '.join(df_anies[df_anies['category'] == 'neutral']['text_en']).split()

# Menghitung jumlah kata pada setiap corpus
corpus_count1 = Counter(corpus1).most_common()
corpus_pos_count1 = Counter(corpus_pos1).most_common()
corpus_neg_count1 = Counter(corpus_neg1).most_common()
corpus_net_count1 = Counter(corpus_net1).most_common()

# Menampilkan informasi
print("Ini merupakan visualisasi untuk data secara keseluruhan tanpa membedakan label")
print("Jumlah kata unik pada data keseluruhan:", len(corpus_count1))
print("Jumlah kata unik pada sentimen positif:", len(corpus_pos_count1))
print("Jumlah kata unik pada sentimen negatif:", len(corpus_neg_count1))
print("Jumlah kata unik pada sentimen netral:", len(corpus_net_count1))

# Menampilkan word clouds
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 7])

wordcloud1 = WordCloud(width=800, height=800, background_color='white', colormap="RdPu", collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_pos1))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Word Cloud Ulasan Positif Anies')

wordcloud2 = WordCloud(width=800, height=800, background_color='white', colormap="Reds",collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_neg1))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Word Cloud Ulasan Negatif Anies')

wordcloud3 = WordCloud(width=800, height=800, background_color='white', colormap="Blues",collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_net1))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Word Cloud Ulasan Netral Anies')
plt.show()

# Menyalin dataframe df
df2 = df

stopwords_prabowo = ['anies', 'cak', 'imin', 'people', 'candidate', 'number', 'presidential', 'indonesia', 'imins',
                   'muhaimin', 'iskandar', 'baswedan', 'prabowo', 'im', 'gibran', 'ganjar', 'reward', 'mahfud',
                   'gangar', 'gangars', 'gibrans', 'md', 'mahfuds', 'mds', 'soebianto', 'subianto', 'rakabuming',
                   'professor', 'bro', 'really', 'milu', 'brother', 'mas', 'peoples', 'pranowo', 'indonesian',
                   'candidates', 'rewards', 'prof', 'already', 'prabowos', 'vice']

stop_words_en.update(stopwords_prabowo)
# Menerapkan remove stopwords
df2['text_en'] = df2['text_en'].apply(remove_stopwords_en)

# Visualisasi word clouds untuk pasangan calon Prabowo-Gibran
# Memfilter paslon atau keyword
df_prabowo = df2[df2['label'] == 'prabowo']

# Membuat corpus untuk kasus all keyword
corpus2 = ' '.join(df_prabowo['text_en']).split()
corpus_pos2 = ' '.join(df_prabowo[df_prabowo['category'] == 'positive']['text_en']).split()
corpus_neg2 = ' '.join(df_prabowo[df_prabowo['category'] == 'negative']['text_en']).split()
corpus_net2 = ' '.join(df_prabowo[df_prabowo['category'] == 'neutral']['text_en']).split()

# Menghitung jumlah kata pada setiap corpus
corpus_count2 = Counter(corpus2).most_common()
corpus_pos_count2 = Counter(corpus_pos2).most_common()
corpus_neg_count2 = Counter(corpus_neg2).most_common()
corpus_net_count2 = Counter(corpus_net2).most_common()

# Menampilkan informasi
print("Ini merupakan visualisasi untuk data secara keseluruhan tanpa membedakan label")
print("Jumlah kata unik pada data keseluruhan:", len(corpus_count2))
print("Jumlah kata unik pada sentimen positif:", len(corpus_pos_count2))
print("Jumlah kata unik pada sentimen negatif:", len(corpus_neg_count2))
print("Jumlah kata unik pada sentimen netral:", len(corpus_net_count2))

# Menampilkan word clouds
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 7])

wordcloud1 = WordCloud(width=800, height=800, background_color='white', colormap="RdPu",collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_pos2))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Word Cloud Ulasan Positif Prabowo')

wordcloud2 = WordCloud(width=800, height=800, background_color='white', colormap="Reds",collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_neg2))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Word Cloud Ulasan Negatif Prabowo')

wordcloud3 = WordCloud(width=800, height=800, background_color='white', colormap="Blues", collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_net2))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Word Cloud Ulasan Netral Prabowo')
plt.show()

# Menyalin dataframe df
df3=df

stopwords_ganjar = ['anies', 'cak', 'imin', 'people', 'candidate', 'number', 'presidential', 'indonesia', 'imins',
                   'muhaimin', 'iskandar', 'baswedan', 'prabowo', 'im', 'gibran', 'ganjar', 'reward', 'mahfud',
                   'gangar', 'gangars', 'gibrans', 'md', 'mahfuds', 'mds', 'soebianto', 'subianto', 'rakabuming',
                   'professor', 'bro', 'really', 'milu', 'brother', 'mas', 'peoples', 'pranowo', 'indonesian',
                   'candidates', 'rewards', 'prof', 'already']

stop_words_en.update(stopwords_ganjar)
# Menerapkan remove stopwords
df3['text_en'] = df3['text_en'].apply(remove_stopwords_en)

# Visualisasi word clouds untuk pasangan calon Ganjar-Mahfud
# Memfilter paslon atau keyword
df_ganjar = df3[df3['label'] == 'ganjar']

# Membuat corpus untuk kasus all keyword
corpus3 = ' '.join(df_ganjar['text_en']).split()
corpus_pos3 = ' '.join(df_ganjar[df_ganjar['category'] == 'positive']['text_en']).split()
corpus_neg3 = ' '.join(df_ganjar[df_ganjar['category'] == 'negative']['text_en']).split()
corpus_net3 = ' '.join(df_ganjar[df_ganjar['category'] == 'neutral']['text_en']).split()

# Menghitung jumlah kata pada setiap corpus
corpus_count3 = Counter(corpus3).most_common()
corpus_pos_count3 = Counter(corpus_pos3).most_common()
corpus_neg_count3 = Counter(corpus_neg3).most_common()
corpus_net_count3 = Counter(corpus_net3).most_common()

# Menampilkan informasi
print("Ini merupakan visualisasi untuk data secara keseluruhan tanpa membedakan label")
print("Jumlah kata unik pada data keseluruhan:", len(corpus_count3))
print("Jumlah kata unik pada sentimen positif:", len(corpus_pos_count3))
print("Jumlah kata unik pada sentimen negatif:", len(corpus_neg_count3))
print("Jumlah kata unik pada sentimen netral:", len(corpus_net_count3))

# Menampilkan word clouds
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 7])

wordcloud1 = WordCloud(width=800, height=800, background_color='white', colormap="RdPu", collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_pos3))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Word Cloud Ulasan Positif Ganjar')

wordcloud2 = WordCloud(width=800, height=800, background_color='white', colormap="Reds",collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_neg3))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Word Cloud Ulasan Negatif Ganjar')

wordcloud3 = WordCloud(width=800, height=800, background_color='white', colormap="Blues", collocations=False, relative_scaling = 0.2,
                       min_font_size=10).generate(' '.join(corpus_net3))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Word Cloud Ulasan Netral Ganjar')
plt.show()

"""Eksplorasi Data Berdasarkan Jumlah Tweet"""

df = pd.read_excel('/content/trans_nostem_nostop.xlsx')

def encode_label(label):
    if label == 'anies':
        return 'paslon 1'
    elif label == 'prabowo':
        return 'paslon 2'
    elif label == 'ganjar':
        return 'paslon 3'
    else:
        return 'lainnya'

df['encoded_label'] = df['label'].apply(encode_label)

# Mendefinisikan fungsi untuk melabel capres-cawapres
def encode_keywords(keyword):
    if keyword in ['ganjar', 'prabowo', 'anies']:
        return 'capres'
    else:
        return 'cawapres'
# Menerapkan fungsi encoded keywords
df['encoded_keywords'] = df['keyword'].apply(encode_keywords)

# Menghitung jumlah tweet untuk masing-masing kategori
tweet_counts = df.groupby(['encoded_label', 'encoded_keywords']).size().unstack(fill_value=0)
tweet_counts = tweet_counts.reindex(columns=['capres', 'cawapres']).fillna(0)

# Menyiapkan data untuk plot
labels = tweet_counts.index
capres_counts = tweet_counts['capres']
cawapres_counts = tweet_counts['cawapres']

# Menampilkan plot
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, capres_counts, width, label='Capres')
rects2 = ax.bar(x + width/2, cawapres_counts, width, label='Cawapres')

# Menambahkan teks label, title dan custom x-axis tick labels
ax.set_xlabel('Nomor Urut Pasangan Calon')
ax.set_ylabel('Jumlah Tweet')
ax.set_title('Jumlah Tweet Kandidat Pasangan Capres-Cawapres')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0.8),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

# Menghitung banyak data berdasarkan kategori sentimen
category_counts = df['encoded_label'].value_counts()

# Mengatur daftar warna yang akan digunakan untuk setiap kategori
colors = ['#3CB371', '#87CEFA', '#B22222']  # Sesuaikan dengan jumlah kategori yang Anda miliki

# Membuat plot batang frekuensi untuk setiap kategori dengan warna yang berbeda
fig, ax = plt.subplots(figsize=(10, 6))
bars = category_counts.plot(kind='bar', color=colors, ax=ax)

# Menambahkan angka di atas setiap batang
for bar in bars.patches:
    ax.annotate(f'{bar.get_height()}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Menambahkan judul dan label sumbu
plt.title('Jumlah Tweet Berdasarkan Pasangan Calon Kandidat', fontsize=16)
plt.xlabel('Nomor urut Kandidat Capres Cawapres', fontsize=14)
plt.ylabel('Jumlah Tweet', fontsize=14)
plt.xticks(rotation=0)
plt.show()

# Menghitung frekuensi untuk masing-masing kategori dan keyword
category_keyword_counts = df.groupby(['sentiment_encoded', 'encoded_label']).size().unstack(fill_value=0)

# Membuat plot batang frekuensi untuk 3 kategori dengan membedakan berdasarkan keyword
plt.figure(figsize=(10, 6))

# Mengatur warna yang akan digunakan untuk setiap keyword
colors = [ 'crimson', 'orange', '#3CB371' ]

# Melakukan looping melalui setiap kategori
for i, (category, data) in enumerate(category_keyword_counts.iterrows()):
    # Plot bar untuk setiap keyword dengan warna yang berbeda
    x = np.arange(len(data))
    plt.bar(x + i * 0.2, data, width=0.2, color=colors[i], label=category)

#Menampilkan plot
plt.title('Jumlah Tweet Berdasarkan Sentimen', fontsize = 20)
plt.xlabel('Nomor Urut Kandidat Capres-Cawapres', fontsize = 14)
plt.ylabel('Jumlah Tweet', fontsize = 14)
plt.xticks(np.arange(len(data)) + 0.2, data.index, rotation=None)
plt.legend(title='Sentimen', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Menghitung frekuensi label untuk keywords 'prabowo'
anies_counts = df[df['keyword'] == 'anies']['sentiment_encoded'].value_counts()

# Menghitung frekuensi label untuk keywords 'gibran'
imin_counts = df[df['keyword'] == 'imin']['sentiment_encoded'].value_counts()

# Menghitung frekuensi label untuk keywords 'prabowo'
prabowo_counts = df[df['keyword'] == 'prabowo']['sentiment_encoded'].value_counts()

# Menghitung frekuensi label untuk keywords 'gibran'
gibran_counts = df[df['keyword'] == 'gibran']['sentiment_encoded'].value_counts()

# Menghitung frekuensi label untuk keywords 'prabowo'
ganjar_counts = df[df['keyword'] == 'ganjar']['sentiment_encoded'].value_counts()

# Menghitung frekuensi label untuk keywords 'gibran'
mahfud_counts = df[df['keyword'] == 'mahfud']['sentiment_encoded'].value_counts()

# Data untuk Pie Chart 'anies'
labels_anies = anies_counts.index
sizes_anies = anies_counts.values

# Data untuk Pie Chart 'cak imin'
labels_imin = imin_counts.index
sizes_imin = imin_counts.values

# Membuat dua subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Warna untuk setiap kategori
colors_sentimen = {'neutral': 'gold', 'positive': '#3CB371', 'negative': 'crimson'}

# Pie Chart untuk 'anies'
axs[0].pie(sizes_anies, labels=labels_anies, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_anies], textprops={'fontsize': 14})
axs[0].set_title('Sentimen Terhadap Anies', fontsize=20)

# Pie Chart untuk 'cak imin'
pie_imin, _, texts_imin = axs[1].pie(sizes_imin, labels=labels_imin, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_imin], textprops={'fontsize': 14})
axs[1].set_title('Sentimen Terhadap Cak Imin', fontsize=20)

#for text in axs[1].texts:
#    text.set_color('white')

#for text in texts_imin:
#    text.set_fontsize(14)

# Menampilkan plot
plt.show()

# Data untuk Pie Chart 'prabowo'
labels_prabowo = prabowo_counts.index
sizes_prabowo = prabowo_counts.values

# Data untuk Pie Chart 'gibran'
labels_gibran = gibran_counts.index
sizes_gibran = gibran_counts.values

# Membuat dua subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Pie Chart untuk 'prabowo'
axs[0].pie(sizes_prabowo, labels=labels_prabowo, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_prabowo], textprops={'fontsize': 14})
axs[0].set_title('Sentimen Terhadap Prabowo', fontsize = 20)

# Pie Chart untuk 'gibran'
axs[1].pie(sizes_gibran, labels=labels_gibran, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_gibran], textprops={'fontsize': 14})
axs[1].set_title('Sentimen Terhadap Gibran', fontsize = 20)

# Menampilkan plot
plt.show()

# Data untuk Pie Chart 'ganjar'
labels_ganjar = ganjar_counts.index
sizes_ganjar = ganjar_counts.values

# Data untuk Pie Chart 'gibran'
labels_mahfud = mahfud_counts.index
sizes_mahfud = mahfud_counts.values

# Membuat dua subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Pie Chart untuk 'ganjar'
axs[0].pie(sizes_ganjar, labels=labels_ganjar, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_ganjar], textprops={'fontsize': 14})
axs[0].set_title('Sentimen Terhadap Ganjar', fontsize = 20)

# Pie Chart untuk 'mahfud'
axs[1].pie(sizes_mahfud, labels=labels_mahfud, autopct='%1.1f%%', startangle=140,
           colors=[colors_sentimen[label] for label in labels_mahfud], textprops={'fontsize': 14})
axs[1].set_title('Sentimen Terhadap Mahfud', fontsize = 20)

# Menampilkan plot
plt.show()

"""Visualisasi n-grams

"""

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

!pip install xlrd
df = pd.read_excel("/content/trans_nostem_sentimen.xlsx")

# Memastikan data teks bertipe string
df['text_en'] = df['text_en'].astype(str)

#Memuat stop words Bahasa Inggris dari library NLTK
nltk.download('stopwords')
# Membuat stopwords bahasa Inggris
stop_words_en = set(stopwords.words('english'))

def remove_stopwords_en(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words_en]
    return ' '.join(filtered_tokens)

# Menerapkan fungsi untuk menghapus stopwords yang telah digabungkan
df['text_en'] = df['text_en'].apply(remove_stopwords_en)

# Mendefinisikan fungsi untuk n-grams
def ngrams(text, n_gram, stop=None):
    if stop is None:
        stop = set()
    tokens = [token for token in text.lower().split(' ') if token != '' if token not in stop]
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

import matplotlib.pyplot as plt
from collections import Counter

def plot_ngram_frequencies(ngram_list, top_n=5, sentiment=None, bar_width=0.5, fig_width=10, fig_height=6, nomor= None):
    ngram_counts = Counter(ngram_list)
    top_ngrams = ngram_counts.most_common(top_n)
    labels, counts = zip(*top_ngrams)

    if nomor == 1:
      color = '#2E8B57'
      candidate = 'Anies - Cak Imin'
    elif nomor == 2:
      color = '#87CEFA'
      candidate = 'Prabowo - Gibran'
    elif nomor == 3:
      color = '#B22222'
      candidate = 'Ganjar - Mahfud'
    else:
      color = 'skyblue'
      candidate = 'Semua Tweet'

    plt.figure(figsize=(fig_width, fig_height))
    plt.barh(labels, counts, color=color, height=bar_width)
    plt.xlabel('Frequency')
    #plt.ylabel('n-gram')
    plt.title('Frekuensi {} {}-gram teratas untuk {}'.format(top_n, len(labels[0].split()), candidate))
    plt.gca().invert_yaxis()
    plt.show()

# Memberikan nilai n pada n-grams yaitu 1,2, dan 3
n = 3
ngrams_result = []

for index, row in df.iterrows():
    keyword = row['label']
    tweet = row['text_en']

    if keyword == 'ganjar': # nilai ini diubah dengan nama kategori capres
        ngrams_result.extend(generate_ngrams(tweet, n))

# Visualisasikan frekuensi n-gram teratas
plot_ngram_frequencies(ngrams_result, fig_width=10, fig_height=3,  nomor=3)

# Mempertimbangkan stopwords tambahan untuk 1/2/3-grams khusus untuk label prabowo
# Menambahkan kata-kata tambahan ke dalam list stopwords
add_stopwords = [ 'pranowo', 'mahfud', 'ganjar', 'md', 'mds', 'ganjars', 'gangar', 'gangars', 'mahfuds',
                 'pranowos', 'candidate', 'serial', 'number', 'presidential', 'vice', 'election', 'candidates',
                  'islamic', 'reward', 'rewards', 'indonesian', 'cak', 'imin', 'prabowo', 'gibran', 'good', 'morning',
                  'viva'
                  ]
stop_words_en.update(add_stopwords)

# Terapkan fungsi untuk menghapus stopwords yang telah digabungkan
df['text_en'] = df['text_en'].apply(remove_stopwords_en)

# Mempertimbangkan stopwords tambahan untuk 1/2/3-grams khusus untuk label prabowo
# Menambahkan kata-kata tambahan ke dalam list stopwords
add_stopwords = [ 'pranowo', 'mahfud', 'ganjar', 'md', 'mds', 'ganjars', 'gangar', 'gangars', 'mahfuds',
                 'pranowos', 'candidate', 'serial', 'number', 'presidential', 'vice', 'election', 'candidates',
                   'reward', 'rewards', 'indonesian', 'cak', 'imin', 'prabowo', 'gibran',  'anies',
                  'prabowo', 'prabowos', 'soebianto', 'gibrans', 'rakabuming', 'raka',  'baswedan', 'muhaimin',
                  'iskandar', 'amen',  'manchester', 'united', 'arsenal', 'god', 'team', 'republic', 'indonesia',
                  'social', 'people', 'really', 'imins', 'im', 'campaign'
                  ]
stop_words_en.update(add_stopwords)

# Terapkan fungsi untuk menghapus stopwords yang telah digabungkan
df['text_en'] = df['text_en'].apply(remove_stopwords_en)

# Mempertimbangkan stopwords tambahan untuk 1/2/3-grams khusus untuk label prabowo
# Menambahkan kata-kata tambahan ke dalam list stopwords
add_stopwords = [ 'pranowo', 'mahfud', 'ganjar', 'md', 'mds', 'ganjars', 'gangar', 'gangars', 'mahfuds',
                 'pranowos', 'candidate', 'serial', 'number', 'presidential', 'vice', 'election', 'candidates',
                  'islamic', 'reward', 'rewards', 'indonesian', 'cak', 'imin', 'prabowo', 'gibran', 'people', 'anies',
                  'prabowo', 'prabowos', 'soebianto', 'gibrans', 'rakabuming', 'raka', 'ethics', 'friends'
                  ]
stop_words_en.update(add_stopwords)

# Menerapkan fungsi untuk menghapus stopwords yang telah digabungkan
df['text_en'] = df['text_en'].apply(remove_stopwords_en)