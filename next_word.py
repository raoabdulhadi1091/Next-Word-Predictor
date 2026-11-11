
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_FILE = "train-00000-of-00001.parquet"
VAL_FILE = "validation-00000-of-00001.parquet"

MAX_WORDS = 12000
SEQUENCE_LENGTH = 8

MAX_TRAIN_SEQUENCES = 250000
MAX_VAL_SEQUENCES = 40000

EPOCHS = 8
BATCH_SIZE = 128

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 2. CHECK DATASET FILES
# ==========================================

print("\nChecking dataset files...")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

if not os.path.exists(VAL_FILE):
    raise FileNotFoundError(f"Missing file: {VAL_FILE}")

print("Training and validation files found!")


# ==========================================
# 3. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 4. LOAD DAILYDIALOG DATASET
# ==========================================

print("\nLoading DailyDialog dataset...")

train_df = pd.read_parquet(TRAIN_FILE)

val_df = pd.read_parquet(VAL_FILE)


def extract_utterances(df):

    sentences = []

    for dialog in df["dialog"]:

        if isinstance(dialog, (list, tuple, np.ndarray)) and len(dialog) > 0:

            sentence = clean_text(dialog[-1])

            if sentence and len(sentence.split()) >= 2:

                sentences.append(sentence)

    return sentences


train_texts = extract_utterances(train_df)

val_texts = extract_utterances(val_df)

del train_df
del val_df

print("Training sentences:", len(train_texts))
print("Validation sentences:", len(val_texts))

if len(train_texts) == 0:
    raise ValueError("Training dataset is empty!")

if len(val_texts) == 0:
    raise ValueError("Validation dataset is empty!")


# ==========================================
# 5. TOKENIZATION
# ==========================================

print("\nPreparing vocabulary...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_texts)

vocab_size = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

print("Vocabulary size:", vocab_size)


# ==========================================
# 6. CREATE TRAINING SEQUENCES
# ==========================================

def create_sequences(sentences, maximum):

    X = []
    y = []

    for sentence in sentences:

        tokens = tokenizer.texts_to_sequences([sentence])[0]

        if len(tokens) < 2:
            continue

        for i in range(1, len(tokens)):

            start = max(0, i - SEQUENCE_LENGTH)

            context = tokens[start:i]

            target = tokens[i]

            padded = pad_sequences(
                [context],
                maxlen=SEQUENCE_LENGTH,
                padding="pre"
            )[0]

            X.append(padded)

            y.append(target)

            if len(X) >= maximum:

                return (
                    np.asarray(X, dtype=np.int32),
                    np.asarray(y, dtype=np.int32)
                )

    return (
        np.asarray(X, dtype=np.int32),
        np.asarray(y, dtype=np.int32)
    )


print("\nCreating training sequences...")

X_train, y_train = create_sequences(
    train_texts,
    MAX_TRAIN_SEQUENCES
)

print("Training sequences:", len(X_train))

print("\nCreating validation sequences...")

X_val, y_val = create_sequences(
    val_texts,
    MAX_VAL_SEQUENCES
)

print("Validation sequences:", len(X_val))


# ==========================================
# 7. VALIDATE TRAINING DATA
# ==========================================

print("\nChecking training data...")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("No training sequences generated!")

if len(X_val) == 0 or len(y_val) == 0:
    raise ValueError("No validation sequences generated!")

if len(X_train) != len(y_train):
    raise ValueError("Training X and y lengths do not match!")

if len(X_val) != len(y_val):
    raise ValueError("Validation X and y lengths do not match!")

print("Dataset checks passed!")


# ==========================================
# 8. BUILD LSTM MODEL
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential([

    Embedding(
        input_dim=vocab_size,
        output_dim=96,
        input_length=SEQUENCE_LENGTH
    ),

    LSTM(128),

    Dropout(0.25),

    Dense(
        vocab_size,
        activation="softmax"
    )

])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\n================================")
print(" MODEL TRAINING STARTED")
print("================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stopping],

    verbose=1

)

print("\nTraining completed!")


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=1
)

print("\nValidation Loss:", round(loss, 4))

print("Validation Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 11. NEXT WORD PREDICTION
# ==========================================

def predict_next_word(sentence):

    sentence = clean_text(sentence)

    if not sentence:
        print("Please enter a valid sentence.")
        return

    tokens = tokenizer.texts_to_sequences([sentence])[0]

    if not tokens:
        print("No recognizable English words found.")
        return

    tokens = tokens[-SEQUENCE_LENGTH:]

    padded = pad_sequences(
        [tokens],
        maxlen=SEQUENCE_LENGTH,
        padding="pre"
    )

    predictions = model.predict(
        padded,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1]

    print("\nTop 5 predicted next words:")

    count = 0

    for index in top_indices:

        word = tokenizer.index_word.get(index)

        if word and word != "<OOV>":

            probability = predictions[index] * 100

            print(
                f"{count + 1}. {word}: "
                f"{probability:.2f}%"
            )

            count += 1

            if count == 5:
                break


# ==========================================
# 12. TEST DIFFERENT SENTENCES
# ==========================================

test_sentences = [

    "how are",

    "how are you",

    "thank you",

    "i love",

    "what do you",

    "would you like",

    "i am going",

    "can you help",

    "good morning",

    "see you"

]

print("\n================================")
print(" TESTING PREDICTIONS")
print("================================")

for sentence in test_sentences:

    print("\nInput:", sentence)

    predict_next_word(sentence)


# ==========================================
# 13. CUSTOM USER INPUT
# ==========================================

print("\n================================")
print(" NEXT WORD PREDICTION SYSTEM")
print("================================")

print("Type exit to stop.")

while True:

    sentence = input("\nEnter your sentence: ")

    if sentence.lower().strip() == "exit":

        print("Program closed!")

        break

    predict_next_word(sentence)
