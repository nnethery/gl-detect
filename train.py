import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
from model import Simple1DCNN
import pandas as pd
import librosa
from tqdm import tqdm

def load_wav_tensor(file_path, target_sample_rate=24000):
    # waveform, sr = torchaudio.load(file_path)
    y, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    if len(y) > 8000:
        start_index = (len(y) - 8000) // 2
        y = y[start_index : start_index + 8000]
    waveform = torch.tensor(y.reshape([1, y.shape[0]]))
    # if sr != target_sample_rate:
    #     waveform = torchaudio.transforms.Resample(sr, target_sample_rate)(waveform)
    return waveform

def preprocess_wav_tensor(waveform, target_length=8000):
    if waveform.size(1) < target_length:
        padding = torch.zeros(1, target_length - waveform.size(1))
        waveform = torch.cat((waveform, padding), 1)
    else:
        waveform = waveform[:, :target_length]
    return waveform

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

df_train = pd.read_csv('data/splits/train.csv')
df_test = pd.read_csv('data/splits/test.csv')
df_val = pd.read_csv('data/splits/val.csv')

df_train = pd.concat([df_train, df_test])

model = Simple1DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
batch_size = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Train Batches", leave=True):
        file_path = row.X
        label = int(row.y)
        waveform = load_wav_tensor(file_path)
        waveform = preprocess_wav_tensor(waveform)
        target = torch.tensor([label])

        optimizer.zero_grad()

        logits = model(waveform.unsqueeze(0))
        
        loss = criterion(logits, target)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(df_train)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')

    model.eval()
    valid_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for i, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Validation Batches", leave=False):
            file_path = row.X
            label = int(row.y)
            waveform = load_wav_tensor(file_path)
            waveform = preprocess_wav_tensor(waveform)
            target = torch.tensor([label])
            output = model(waveform.unsqueeze(0))
            loss = criterion(output, target)
            valid_loss += loss.item()

            prediction = torch.argmax(output, dim=1)
            if prediction.item() == label:
                correct_predictions += 1

    valid_loss /= len(df_val)
    val_accuracy = correct_predictions / len(df_val)
    print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}")
    print(f'Validation accuracy: {val_accuracy}')

save_checkpoint(model, optimizer, num_epochs, total_loss, "checkpoint.pth")