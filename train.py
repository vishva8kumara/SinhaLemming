
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

# # Encoder
# class Encoder(nn.Module):
#     def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=3):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         outputs, (h, c) = self.lstm(embedded)
#         return outputs, (h, c)

# # Attention
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Linear(hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))

#     def forward(self, hidden, encoder_outputs):
#         batch_size = encoder_outputs.size(0)
#         seq_len = encoder_outputs.size(1)

#         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         energy = energy.transpose(1, 2)
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)
#         attn_weights = torch.bmm(v, energy).squeeze(1)

#         return torch.softmax(attn_weights, dim=1)

# # Decoder
# class Decoder(nn.Module):
#     def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=3):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.attention = Attention(hidden_size)
#         self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, input_token, hidden, cell, encoder_outputs):
#         embedded = self.embedding(input_token).unsqueeze(1)
#         attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
#         context = torch.bmm(attn_weights, encoder_outputs)
#         rnn_input = torch.cat((embedded, context), dim=2)

#         output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
#         output = self.fc(output.squeeze(1))
#         return output, hidden, cell

# # Full Model
# class Seq2Seq(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.encoder = Encoder(vocab_size)
#         self.decoder = Decoder(vocab_size)

#     def forward(self, src, tgt):
#         batch_size = src.size(0)
#         tgt_len = tgt.size(1)
#         vocab_size = self.decoder.fc.out_features

#         outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(device)
#         encoder_outputs, (hidden, cell) = self.encoder(src)

#         input_token = tgt[:, 0]

#         for t in range(1, tgt_len):
#             output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
#             outputs[:, t] = output
#             input_token = tgt[:, t-1]

#         return outputs

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
