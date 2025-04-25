# Sinhala Lemmatizer - Seq2Seq (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json

with open('input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Sample Training Data
data_pairs = list(data.items())

with open("mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)

char2idx = mappings["char2idx"]
idx2char = {int(k): v for k, v in mappings["idx2char"].items()}	# Convert idx2char keys back to int
vocab_size = mappings["vocab_size"]
MAX_LEN = 20

# Dataset
class LemmaDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def encode(self, word):
        return [char2idx.get(c, 0) for c in word] + [0] * (MAX_LEN - len(word))

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return torch.tensor(self.encode(x)), torch.tensor(self.encode(y))

    def __len__(self):
        return len(self.pairs)

# Encoder-Decoder Model
class LemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.encoder = nn.LSTM(64, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.decoder = nn.LSTM(64, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x, y):
        x = self.embedding(x)
        y = self.embedding(y)
        _, (h, c) = self.encoder(x)
        out, _ = self.decoder(y, (h, c))
        return self.fc(out)

# Training Loop
def train(model, dataloader, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    lowest_loss = 10000

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if True: #epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
            if lowest_loss > total_loss:
                torch.save(model.state_dict(), 'sinhalemming.pth')
                lowest_loss = total_loss

# Run Everything
dataset = LemmaDataset(data_pairs)
dataloader = DataLoader(dataset, batch_size=34, shuffle=True)

model = LemmaModel()
train(model, dataloader)

print('Completed')
