from collections import Counter, defaultdict
import re

# === YOUR SLIDE DATASET ===
dataset = [
    "The cat sat on the mat.",
    "The cat is sleeping.",
    "The cat chased the mouse.",
    "The cat is playing with a ball.",
    "The fish swam in the pond.",
    "The fish is colorful in a bowl.",
    "The fish is in the aquarium.",
    "The dog is barking at the mall.",
    "The bird sang in the morning.",
    "The horse ran across the field."
]

# === BUILD MARKOV BIGRAM MODEL ===
all_words = []
for sentence in dataset:
    words = re.findall(r'\w+', sentence.lower())
    all_words.extend(words)

# Learn bigrams: P(next|current)
bigram_model = defaultdict(list)
for i in range(len(all_words)-1):
    bigram_model[all_words[i]].append(all_words[i+1])

# === PREDICTION FUNCTION ===
def predict_next_word(context):
    if context in bigram_model:
        candidates = bigram_model[context]
        counts = Counter(candidates)
        total = sum(counts.values())
        probs = {word: count/total for word, count in counts.items()}
        best = max(probs, key=probs.get)
        return best, probs
    return "unknown", {}

# === DEMO: EXACTLY LIKE YOUR SLIDE ===
print("=== MARKOV BIGRAM MODEL PREDICTIONS ===\n")
cat_pred, cat_probs = predict_next_word("cat")
fish_pred, fish_probs = predict_next_word("fish")

print(f"'The cat ___?' → **{cat_pred}**")
print("Probabilities:", {k: f"{v:.2f}" for k,v in cat_probs.items()})
print()
print(f"'The fish ___?' → **{fish_pred}**")
print("Probabilities:", {k: f"{v:.2f}" for k,v in fish_probs.items()})

# === MODEL STATS ===
print("\n=== MODEL STATS ===")
print(f"Vocabulary size: {len(set(all_words))}")
print(f"Bigrams learned: {sum(len(words) for words in bigram_model.values())}")
print("\nDataset patterns learned:")
for context in ["cat", "fish", "the"]:
    pred, probs = predict_next_word(context)
    print(f"{context} → {pred}")
