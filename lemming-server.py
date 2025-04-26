
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
