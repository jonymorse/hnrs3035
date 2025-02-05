import random
from collections import defaultdict, Counter
import pickle
import argparse

class NgramModel:
    def __init__(self, n):
        self.n = n  # Order of the n-gram (1 for unigram, 2 for bigram)
        self.vocab = set()  # Unique words in the corpus
        self.ngrams = defaultdict(Counter)  # Stores n-grams and their counts

    def train(self, data):
        # Split the data into words (treat punctuation as separate words)
        words = data.split()
        self.vocab = set(words)

        # Generate n-grams and count occurrences
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n])
            next_word = words[i + self.n] if i + self.n < len(words) else None
            if next_word:
                self.ngrams[ngram][next_word] += 1

    def predict_next_word(self, input_words, deterministic=False):
        input_words = tuple(input_words)
        if input_words not in self.ngrams:
            return "Error: Input word(s) not found in vocabulary."

        next_word_counts = self.ngrams[input_words]
        total = sum(next_word_counts.values())
        probabilities = {word: count / total for word, count in next_word_counts.items()}

        if deterministic:
            return max(probabilities, key=probabilities.get)
        else:
            return random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocabulary = set()

    def train(self, corpus, k=500):
        # Preprocess the corpus
        words = re.findall(r'\w+|\S', corpus)
        vocab = Counter(words)

        # Initialize vocabulary with characters
        self.vocabulary = set()
        for word in vocab:
            self.vocabulary.update(list(word))

        # BPE algorithm
        for _ in range(k):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = list(word)
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            self.vocabulary.add(new_token)

            # Update the vocabulary
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = word.replace(''.join(best_pair), new_token)
                new_vocab[new_word] = freq
            vocab = new_vocab

    def tokenize(self, text):
        tokens = []
        token_ids = []
        current_text = text

        while current_text:
            for token in sorted(self.vocabulary, key=len, reverse=True):
                if current_text.startswith(token):
                    tokens.append(token)
                    token_ids.append(hash(token) % 1000)  # Simple hash for ID
                    current_text = current_text[len(token):]
                    break
            else:
                tokens.append(current_text[0])
                token_ids.append(hash(current_text[0]) % 1000)
                current_text = current_text[1:]

        return tokens, token_ids

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
        
def main():
    parser = argparse.ArgumentParser(description="N-gram Model and BPE Tokenizer")
    parser.add_argument('activity', choices=['train_ngram', 'predict_ngram', 'train_bpe', 'tokenize'],
                        help="Activity to perform")
    parser.add_argument('--data', help="Path to training data corpus")
    parser.add_argument('--save', help="Path to save the model")
    parser.add_argument('--load', help="Path to load the model")
    parser.add_argument('--word', help="Input word(s) for prediction")
    parser.add_argument('--nwords', type=int, help="Number of words to predict")
    parser.add_argument('--text', help="Text to tokenize")
    parser.add_argument('--n', type=int, choices=[1, 2], help="Order of the n-gram")
    parser.add_argument('--d', action='store_true', help="Deterministic flag for prediction")
    args = parser.parse_args()

    if args.activity == 'train_ngram':
        with open(args.data, 'r') as f:
            data = f.read()
        model = NgramModel(args.n)
        model.train(data)
        model.save_model(args.save)
        print(f"N-gram model trained and saved to {args.save}")

    elif args.activity == 'predict_ngram':
        model = NgramModel.load_model(args.load)
        input_words = args.word.split()
        for _ in range(args.nwords):
            next_word = model.predict_next_word(input_words, deterministic=args.d)
            print(next_word, end=' ')
            input_words = input_words[1:] + [next_word]
        print()

    elif args.activity == 'train_bpe':
        with open(args.data, 'r') as f:
            data = f.read()
        tokenizer = BPETokenizer()
        tokenizer.train(data)
        tokenizer.save_model(args.save)
        print(f"BPE tokenizer trained and saved to {args.save}")

    elif args.activity == 'tokenize':
        tokenizer = BPETokenizer.load_model(args.load)
        tokens, token_ids = tokenizer.tokenize(args.text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)

if __name__ == "__main__":
    main()