
from flask import Flask, Response
from sinhalemming import Encoder, Attention, Decoder, Seq2Seq
import urllib.parse
import json

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load mappings
with open("mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)

char2idx = mappings["char2idx"]
idx2char = {int(k): v for k, v in mappings["idx2char"].items()}
vocab_size = mappings["vocab_size"]
MAX_LEN = 20

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

# Load Model
model = Seq2Seq(vocab_size)
model.load_state_dict(torch.load("sinhalemming.pth", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict(model, word, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        # Encode input word
        encoded = [char2idx.get(c, char2idx['?']) for c in word] + [0] * (max_len - len(word))
        src = torch.tensor([encoded], dtype=torch.long).to(device)
        encoder_outputs, (hidden, cell) = model.encoder(src)

        input_token = torch.tensor([char2idx['<']]).to(device)
        decoded = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred = output.argmax(dim=-1).item()

            if idx2char[pred] == '>':
                break
            decoded.append(idx2char[pred])
            input_token = torch.tensor([pred]).to(device)

        return ''.join(decoded)

# Flask endpoint
@app.route('/<path:word>')
def predict_endpoint(word):
    # Decode URL-encoded word
    decoded_word = urllib.parse.unquote(word)
    try:
        result = predict(model, '<'+decoded_word)
        return Response(result, mimetype='text/plain; charset=utf-8')
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
