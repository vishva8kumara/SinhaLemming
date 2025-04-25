# Sinhala Lemmatizer

This project implements a sequence-to-sequence (Seq2Seq) LSTM model for lemmatizing Sinhala words, mapping inflected forms to their base forms (lemmas). The model is built using PyTorch and trained on a dataset of Sinhala word-lemma pairs. It supports the Sinhala script and handles morphological variations common in the language.

## Features
- **Seq2Seq Architecture**: Uses an encoder-decoder LSTM with embeddings for character-level lemmatization.
- **Sinhala Support**: Handles Sinhala Unicode characters, including vowels, consonants, and diacritics.
- **Training and Inference**: Includes scripts for training (`train.py`) and testing (`test.py`) the model.
- **Custom Vocabulary**: Maps Sinhala characters to indices (`mappings.json`) for robust encoding.

## Dataset
The model is trained on `input.json`, which contains pairs of inflected Sinhala words and their lemmas. Example:
```json
{
  "අකුරු": "<අකුර>",
  "අගයන්": "<අගය>",
  "ඈත": "<ඈත>",
  ...
}
```
Lemmas are wrapped in angle brackets (`<`, `>`) to distinguish them from input words.

## Requirements
- Python 3.8+
- PyTorch (`pip install torch`)
- JSON for data handling (standard library)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sinhala-lemmatizer.git
   cd sinhala-lemmatizer
   ```
2. Install dependencies:
   ```bash
   pip install torch
   ```
3. Ensure `input.json` and `mappings.json` are in the project directory.

## Usage
### Training
To train the model:
```bash
python train.py
```
- Loads `input.json` and `mappings.json`.
- Trains a Seq2Seq LSTM model with a two-layer encoder-decoder architecture.
- Saves the best model weights to `sinhalemming.pth` based on validation loss.

### Testing
To test the model on sample words:
```bash
python test.py
```
- Loads the trained model (`sinhalemming.pth`) and `mappings.json`.
- Predicts lemmas for test words, e.g.:
  ```
  Starting Predict
  අක්මාවේ → <අක්මා>
  අගයනවා → <අගය>
  ඈත → <ඈත>
  ...
  ```

### Customizing
- **Add Words**: Update `input.json` with new word-lemma pairs.
- **Extend Vocabulary**: Modify `mappings.json` or regenerate it using the `create_mappings` function in `train.py` to include additional Sinhala characters.
- **Hyperparameters**: Adjust `train.py` (e.g., batch size, learning rate, LSTM layers) for better performance.

## Files
- `train.py`: Script for training the lemmatizer model.
- `test.py`: Script for testing the model on sample words.
- `mappings.json`: Character-to-index mappings for Sinhala characters and special tokens.
- `input.json`: Training dataset with word-lemma pairs.
- `sinhalemming.pth`: Trained model weights (generated after training).

## License
This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. You are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution**: You must give appropriate credit to the original author, provide a link to the license, and indicate if changes were made.
- **No Additional Restrictions**: You may not apply legal terms or technological measures that restrict others from doing anything the license permits.

See the [full license](https://creativecommons.org/licenses/by/4.0/) for details.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Contact
For questions or suggestions, open an issue on GitHub or contact [your-email@example.com].

## Acknowledgments
- Built with [PyTorch](https://pytorch.org/).
- Inspired by Seq2Seq models for natural language processing.