import pandas as pd
import numpy as np
!pip install xlrd

df = pd.read_excel('/content/trans_nostem_nostop.xlsx')
df = df[['text_in', 'text_en', 'keyword', 'label', 'sentiment_encoded']]

import torch
from tqdm.notebook import tqdm

df.sentiment_encoded.value_counts()
possible_labels = df.sentiment_encoded.unique()
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
  label_dict[possible_labels] = index
df["category"] = df.sentiment_encoded.replace(label_dict)

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(df.index.values, df.category.values, test_size=0.20, random_state=42, stratify=df.category.values)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'training'
df.loc[X_val, 'data_type'] = 'validation'
df.loc[X_test, 'data_type'] = 'test'
df.groupby(['sentiment_encoded', 'label', 'data_type']).count()

# Tokenisasi
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

df['text_en'] = df['text_en'].astype(str)
df['text_en'] = df['text_en'].fillna('')

train_encode = tokenizer.batch_encode_plus(
    df[df['data_type'] == 'training']['text_en'].values,
    pad_to_max_length=True,
    return_attention_mask=True,
    add_special_tokens=True,
    max_length=None,
    return_tensors='pt'
)

valid_encode = tokenizer.batch_encode_plus(
    df[df['data_type'] == 'validation']['text_en'].values,
    pad_to_max_length=True,
    return_attention_mask=True,
    add_special_tokens=True,
    max_length=None,
    return_tensors='pt'
)

test_encode = tokenizer.batch_encode_plus(
    df[df['data_type'] == 'test']['text_en'].values,
    pad_to_max_length=True,
    return_attention_mask=True,
    add_special_tokens=True,
    max_length=None,
    return_tensors='pt'
)

train_input = train_encode['input_ids']
train_attention = train_encode['attention_mask']
train_labels = torch.tensor(df[df.data_type == 'training'].category.values)

valid_input = valid_encode['input_ids']
valid_attention = valid_encode['attention_mask']
valid_labels = torch.tensor(df[df.data_type == 'validation'].category.values)

test_input = test_encode['input_ids']
test_attention = test_encode['attention_mask']
test_labels = torch.tensor(df[df.data_type == 'test'].category.values)

train_data = TensorDataset(train_input,
                           train_attention,
                           train_labels)
valid_data = TensorDataset(valid_input,
                           valid_attention,
                           valid_labels)
test_data = TensorDataset(test_input,
                          test_attention,
                          test_labels)

print(len(train_data), len(valid_data), len(test_data))

# Model Pre-trained BERT
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = len(label_dict), output_attentions=False, output_hidden_states=False)

# Membuat data loader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 16
train_dataloader = DataLoader(train_data, sampler = RandomSampler(train_data), batch_size = BATCH_SIZE)
valid_dataloader = DataLoader(valid_data, sampler = SequentialSampler(valid_data), batch_size = BATCH_SIZE)
test_dataloader = DataLoader(test_data, sampler = SequentialSampler(test_data), batch_size = BATCH_SIZE)

# Optimizer dan Scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-1)
EPOCHS = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=len(train_dataloader)*EPOCHS)

# Metrik Evaluasi
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def accuracy(preds, labels):
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return accuracy_score(labels_flat, preds_flat)

def f1(preds, labels):
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return f1_score(labels_flat, preds_flat, average='weighted')

def precision(preds, labels):
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return precision_score(labels_flat, preds_flat, average='weighted')

def recall(preds, labels):
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return recall_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(valid_dataloader):
  model.eval()

  total_eval_loss = 0
  y_hat, y = [], []

  for batch in valid_dataloader:
      batch = tuple(b.to(device) for b in batch)
      inputs = {'input_ids' : batch[0],
                'attention_mask': batch[1],
                'labels' : batch[2]
                }

      with torch.no_grad():
        outputs = model(**inputs)

      loss = outputs[0]
      logits = outputs[1]
      total_eval_loss += loss.item()

      logits = logits.detach().cpu().numpy()
      label_ids = inputs['labels'].cpu().numpy()
      y_hat.append(logits)
      y.append(label_ids)

  avg_eval_loss = total_eval_loss/len(valid_dataloader)

  y_hat = np.concatenate(y_hat, axis=0)
  y = np.concatenate(y, axis=0)

  return avg_eval_loss, y_hat, y

import matplotlib.pyplot as plt

training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
f1_scores = []

for epoch in tqdm(range(1, EPOCHS+1)):
    model.train()

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
                  }
        outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        # Menghitung akurasi pada pelatihan
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        labels = inputs['labels'].cpu().numpy()
        correct_predictions += np.sum(preds == labels)
        total_predictions += len(labels)

    torch.save(model.state_dict(), f'BERT_semantic_epoch_{epoch}.pt')

    tqdm.write(f'\nEpoch {epoch}')

    avg_training_loss = total_loss / len(train_dataloader)
    training_losses.append(avg_training_loss)
    training_accuracy = correct_predictions / total_predictions
    training_accuracies.append(training_accuracy)
    tqdm.write(f'Training loss: {avg_training_loss}')
    tqdm.write(f'Training accuracy: {training_accuracy}')

    val_loss, predictions, actual = evaluate(valid_dataloader)
    validation_losses.append(val_loss)
    val_accuracy = np.sum(np.argmax(predictions, axis=1) == actual) / len(actual)
    validation_accuracies.append(val_accuracy)
    score_f1 = f1(predictions, actual)
    f1_scores.append(score_f1)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Validation accuracy: {val_accuracy}')
    tqdm.write(f'F1 Score (Weighted): {score_f1}')

# Plot loss dari data training and validation
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), training_losses, label='Training Loss')
plt.plot(range(1, EPOCHS+1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot akurasi dari data training and validation
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), training_accuracies, label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot F1 score
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), f1_scores, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.show()

# Load model yang sudah dilatih
model.load_state_dict(torch.load('/content/BERT_semantic_epoch_4.pt', map_location=torch.device('cpu')))

# Evaluasi model dengan data testing
test_loss, predictions, actual = evaluate(test_dataloader)
test_precision = precision(predictions, actual)
test_recall = recall(predictions, actual)
test_accuracy = accuracy(predictions, actual)
test_f1 = f1(predictions, actual)
test_acc_per_class = accuracy_per_class(predictions, actual)

tqdm.write(f'\nTest loss: {test_loss}')
tqdm.write(f'\nAccuracy : {test_accuracy}')
tqdm.write(f'\nPrecision : {test_precision}')
tqdm.write(f'\nRecall loss: {test_recall}')
tqdm.write(f'\nF1 score loss: {test_f1}')

#Memeriksa label sentimen
label_dict

tqdm.write(f'\nValidation Accuracy per class: {val_acc_per_class}')

# Confusion matrix dan classification report
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

predictions_1 = np.argmax(predictions, axis=1).flatten()
actual_1 = actual.flatten()

# Menyimpan label predictions dan actual
np.save('predictions.npy', predictions_1)
np.save('actual.npy', actual_1)

# Load label predictions dan actual
predictions_1 = np.load('predictions.npy')
actual_1 = np.load('actual.npy')

# Membuat label disesuaikan dengan label_dict
label_dict = {0: 'negatif', 1: 'netral', 2: 'positif'}

# Classification report
print("Classification Report:")
print(classification_report(actual_1, predictions_1, target_names=label_dict.values()))

# Confusion matrix
conf_matrix = confusion_matrix(actual_1, predictions_1)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Test Dataset')
plt.show()