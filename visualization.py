import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords

# Load the data
file_path = r'C:\Users\kacha\Downloads\Prodigy\Task_4\Reviews.csv'  
df = pd.read_csv(file_path)

# Define a function to clean the review text
def clean_text(text):
    # Remove punctuation and numbers, convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply the cleaning function to the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# 1-2 = Negative, 3 = Neutral, 4-5 = Positive
df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else ('Neutral' if x == 3 else 'Negative'))

# Plot the distribution of sentiment
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

print(df[['Cleaned_Text', 'Sentiment']].head())
