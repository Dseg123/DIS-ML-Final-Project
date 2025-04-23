import kagglehub
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset

df = pd.read_csv('sample_reviews.csv')

# Encode the target variable 'label' into numerical format
df['label_encoded'] = df['label'].apply(lambda x: 1 if x == 'CG' else 0)


# Function to clean text (basic cleaning, but keep punctuation for embeddings)
def clean_text2(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()  # Convert to lowercase

# Clean text_
df["text_"] = df["text_"].astype(str).apply(clean_text2)



# Step 1: Prepare data
train_dataset = Dataset.from_pandas(df[["text_", "label_encoded"]])
train_examples = [
    InputExample(texts=[row["text_"], row["text_"]], label=int(row["label_encoded"]))
    for _, row in df.iterrows()
]

# Step 2: Build model
my_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Training setup
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.SoftmaxLoss(
    model=my_model,
    sentence_embedding_dimension=my_model.get_sentence_embedding_dimension(),
    num_labels=2
)

# Step 4: Train
my_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100
)

# Step 5: Save or use the model
my_model.save("finetuned-embedding-model/")