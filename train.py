
# Sinhala Lemmatizer - Seq2Seq with Attention (PyTorch)

from torch.utils.data import Dataset, DataLoader
from sinhalemming import Encoder, Attention, Decoder, Seq2Seq
import torch
import torch.nn as nn
import random
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data_pairs = list(data.items())

# Load mappings
with open("mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)

char2idx = mappings["char2idx"]
idx2char = {int(k): v for k, v in mappings["idx2char"].items()}
vocab_size = mappings["vocab_size"]
MAX_LEN = 20

# Dataset
class LemmaDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def encode(self, word):
        return [char2idx.get(c, char2idx['?']) for c in word] + [0] * (MAX_LEN - len(word))

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return torch.tensor(self.encode(x)), torch.tensor(self.encode(y))

    def __len__(self):
        return len(self.pairs)

# Validation
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training Loop
def train(model, train_loader, val_loader, epochs=200, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0035, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    lowest_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch}, Train Loss: {total_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), 'sinhalemming.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# Main
if __name__ == "__main__":
    random.shuffle(data_pairs)
    train_size = int(0.8 * len(data_pairs))
    train_pairs, val_pairs = data_pairs[:train_size], data_pairs[train_size:]

    train_dataset = LemmaDataset(train_pairs)
    val_dataset = LemmaDataset(val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Seq2Seq(vocab_size).to(device)
    train(model, train_loader, val_loader)

print('Completed')
