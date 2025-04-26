
import json
import torch
from sinhalemming import Encoder, Attention, Decoder, Seq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load mappings
with open("mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)

char2idx = mappings["char2idx"]
idx2char = {int(k): v for k, v in mappings["idx2char"].items()}
vocab_size = mappings["vocab_size"]
MAX_LEN = 20

# Load Model
model = Seq2Seq(vocab_size)
model.load_state_dict(torch.load("sinhalemming.pth", map_location=device))
model.to(device)
model.eval()

# Predict
def predict(model, word, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        encoded = [char2idx.get(c, char2idx['?']) for c in word] + [0] * (max_len - len(word))# + 3
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

# Sample prediction
test_words = ["අක්මාවේ", "අගයනවා", "අක්‍රමිකතාව", "අක්‍රමිකතාවයන්", "අගයෙන්", "අගමැතිගෙන්", "අක්ෂරය", "තත්වාකාරයෙන්",
    "තත්වාකාරය", "තරුණයෙකුගේ", "ඔලෙයොරෙසින්", "ඔලෙයොරෙසින", "ඕසෝනගෝලය", "ඕසෝනගෝලයේ", "කඩිනම්ගතිය",
    "අඟහරුවාදා", "ශුෂ්කාක්ෂිරෝගය", "අජටාකාශවිද්‍යාව", "කමිටුවේදීයි", "කමිටුවක්"]

print("Starting Predict")
for w in test_words:
    print(f"{w} → {predict(model, '<'+w)}")
