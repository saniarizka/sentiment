import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

!pip install xlrd

df = pd.read_excel("/content/trans_nostem_nostop.xlsx")

df = df[['text_in', 'text_en', 'keyword', 'label', 'sentiment_encoded']]
col_names = ['text_in', 'text_en', 'keyword', 'label', 'sentiment_encoded']
df.columns = col_names
df.head()

possible_labels = df.sentiment_encoded.unique()
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
  label_dict[possible_labels] = index
df["category"] = df.sentiment_encoded.replace(label_dict)

# Mengganti NaN dengan string kosong
df['text_en'] = df['text_en'].fillna('')

# Memastikan semua elemen adalah string
df['text_en'] = df['text_en'].astype(str)

label = np.array(df['category'], dtype ='float64')

# Memastikan data input berupa string
df['text_en'] = df['text_en'].astype(str)

# Menggunakan Keras untuk tokenize dan pad input
tokenizer = Tokenizer(num_words = 12500, oov_token="OOV")
tokenizer.fit_on_texts(df['text_en'])
word_index = tokenizer.word_index

# Mengonversi teks menjadi urutan token
train = tokenizer.texts_to_sequences(df['text_en'])

# Melakukan padding pada urutan token dengan jenis padding "pos"
data = pad_sequences(train, padding="post")

# Membagi dataset menjadi data training, validation dan test
X_train_val, X_test, y_train_val, y_test = train_test_split(data, label, test_size=0.2, random_state=42,stratify=df.category.values)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)

# Menampilkan hasil splitting dataset
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

# Membetuk model Bi-LSTM
embedding_dim =300
model = Sequential()
model.add(Embedding(input_dim=12500, output_dim=embedding_dim, input_length=maxlen))
model.add(Bidirectional(LSTM(32, activation = 'tanh')))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))
model.summary()

# Melakukan kompilasi model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-5),
    metrics=['accuracy']
)

# Pelatihan model
history = model.fit(
    X_train,
    y_train,
    epochs=4,
    batch_size=16,
    validation_data=(X_val, y_val)
)

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Melakukan prediksi terhadap data testing
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Melakukan evaluasi model dengan data test
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,  average='weighted')
recall = recall_score(y_test, y_pred,  average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrik evaluasi
print(f'Accuracy: {accuracy*100}%')
print(f'Precision: {precision*100}%')
print(f'Recall: {recall*100}%')
print(f'F1 Score: {f1*100}%')

from sklearn.metrics import classification_report

# Menampilkan classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Mengonversi hasil prediksi to label kelas
predictions = np.argmax(y_pred_prob, axis=1).flatten()
actual = y_test.flatten()

# Menyimpan label predictions and actual
np.save('predictions.npy', predictions)
np.save('actual.npy', actual)

# Load label predictions and actual
predictions_1 = np.load('predictions.npy')
actual_1 = np.load('actual.npy')

# Label dictionary (disesuaikan dengan data)
label_dict = {0: 'negatif', 1: 'netral', 2: 'positif'}

# Confusion matrix
conf_matrix = confusion_matrix(actual_1, predictions_1)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Test Dataset')
plt.show()

import matplotlib.pyplot as plt

# Plot akurasi pada proses pelatihan
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss pada proses pelatihan
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()