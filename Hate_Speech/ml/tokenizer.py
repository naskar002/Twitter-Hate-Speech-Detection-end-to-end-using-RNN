from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

class Tokenizer():
    def __init__(self, num_words):
        self.num_words = num_words
        self.stoi = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
        self.itos = {0: "<PAD>", 1: "<UNK>"}  # Reverse mapping

    def fit_on_texts(self, texts):
        """Build vocabulary from training data only."""
        tokenized_texts = [word_tokenize(text) for text in texts]  # Tokenize each sentence
        counter = Counter(word for txt in tokenized_texts for word in txt)  # Count word occurrences

        # Keep most frequent words (excluding special tokens)
        sorted_vocab = sorted(counter.items(), key=lambda x: x[1], reverse=True)[: self.num_words - 2]

        # Assign indices starting from 2 (0 and 1 are reserved)
        for idx, (word, _) in enumerate(sorted_vocab, start=2):
            self.stoi[word] = idx
            self.itos[idx] = word

    def text_to_sequences(self, texts):
        """Convert text to sequences using the same vocabulary."""
        tokenized_texts = [word_tokenize(text) for text in texts]  # Tokenize each sentence
        sequences = [[self.stoi.get(word, self.stoi["<UNK>"]) for word in txt] for txt in tokenized_texts]
        return sequences

    def pad_sequences(self, sequences, maxlen):
        """Manually pad sequences to fixed maxlen."""
        padded_seqs = [
            [self.stoi["<PAD>"]] * (maxlen - len(seq)) +seq  if len(seq) < maxlen else seq[:maxlen]
            for seq in sequences
        ]
        return torch.tensor(padded_seqs, dtype=torch.int64)

