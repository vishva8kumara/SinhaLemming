
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (h, c) = self.lstm(embedded)
        return outputs, (h, c)

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = 32 #hidden_size ** 0.5  # scaling factor to soften attention

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        hidden = hidden.unsqueeze(1)  # (batch, 1, hidden)
        attn_scores = torch.bmm(hidden, encoder_outputs.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        attn_scores = attn_scores / self.scale  # soften by scale
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        return attn_weights

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)#, bidirectional=False
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embed_size)

        attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_size)

        rnn_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed + hidden)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = self.fc(output.squeeze(1))  # (batch, vocab_size)

        return output, hidden, cell

# Full Model
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def forward(self, src, tgt):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(device)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            input_token = tgt[:, t-1]

        return outputs
