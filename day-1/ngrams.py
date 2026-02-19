import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter

# Dataset (Corpus)
corpus = [
    "N-grams enhance language processing tasks.",
    "Language models predict the next word.",
    "N-grams are useful for prediction.",
    "Language processing is important."
]

# Tokenize and combine all sentences
tokens = []
for sentence in corpus:
    tokens.extend(word_tokenize(sentence.lower()))

# ------------------------
# Build Bigram Model
# ------------------------
bigram_model = defaultdict(Counter)

for w1, w2 in ngrams(tokens, 2):
    bigram_model[w1][w2] += 1

# ------------------------
# Build Trigram Model
# ------------------------
trigram_model = defaultdict(Counter)

for w1, w2, w3 in ngrams(tokens, 3):
    trigram_model[(w1, w2)][w3] += 1


# ------------------------
# Prediction Functions
# ------------------------

def predict_bigram(word):
    return bigram_model[word].most_common(1)

def predict_trigram(word1, word2):
    return trigram_model[(word1, word2)].most_common(1)


# Example Predictions
print("Bigram prediction for 'language':", predict_bigram("language"))
print("Trigram prediction for ('language', 'processing'):", predict_trigram("language", "processing"))
