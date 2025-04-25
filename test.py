import torch
from torch import nn
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)

char2idx = mappings["char2idx"]
idx2char = {int(k): v for k, v in mappings["idx2char"].items()}	# Convert idx2char keys back to int
vocab_size = mappings["vocab_size"]
MAX_LEN = 20

# Define the same model architecture you used for training
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

# Load model and weights
model = LemmaModel()
model.load_state_dict(torch.load("sinhalemming.pth", map_location=device))
model.eval()

# Inference
def predict_one(model, word):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[char2idx.get(c, 0) for c in word] + [0]*(MAX_LEN - len(word))])
        y = torch.zeros((1, MAX_LEN), dtype=torch.long)
        output = model(x, y)
        pred = output.argmax(2).squeeze().tolist()
        chars = [idx2char[i] for i in pred if i != 0]
        return ''.join(chars)

def predict(model, word, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        # Encode input word
        encoded = [char2idx.get(c, char2idx['?']) for c in word] + [0] * (max_len - len(word))
        src = torch.tensor([encoded], dtype=torch.long).to(device)
        
        # Encoder
        embedded = model.embedding(src)
        _, (h, c) = model.encoder(embedded)
        
        # Decoder: Start with '<'
        output = [char2idx['<']]
        decoded = []
        
        for _ in range(max_len):
            y = torch.tensor([output[-1]], dtype=torch.long).unsqueeze(0).to(device)
            y_embed = model.embedding(y)
            out, (h, c) = model.decoder(y_embed, (h, c))
            logits = model.fc(out.squeeze(1))
            pred = torch.argmax(logits, dim=-1).item()
            
            if pred == char2idx['>']:  # Stop at '>'
                #decoded.append(idx2char[pred])
                break
            decoded.append(idx2char[pred])
            output.append(pred)
        
        return ''.join(decoded)

# Sample prediction
test_words = ["අක්මාවේ", "අගයනවා", "අක්‍රමිකතාව", "අක්‍රමිකතාවයන්", "අගයෙන්", "අගමැතිගෙන්", "අක්ෂරය", "තත්වාකාරයෙන්",
		"තත්වාකාරය", "තරුණයෙකුගේ", "ඔලෙයොරෙසින්", "ඔලෙයොරෙසින", "ඕසෝනගෝලය", "ඕසෝනගෝලයේ", "කඩිනම්ගතිය",
		"අඟහරුවාදා", "ශුෂ්කාක්ෂිරෝගය", "අජටාකාශවිද්‍යාව", "කමිටුවේදීයි", "කමිටුවක්"]
print("Starting Predict")
for w in test_words:
    print(f"{w} → {predict(model, '<'+w)}")
