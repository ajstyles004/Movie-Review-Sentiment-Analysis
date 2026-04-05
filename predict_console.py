import joblib
import re
import warnings
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import os

warnings.filterwarnings('ignore')

print("Initializing console...")

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Remove HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    # 2. Keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Lowercase
    text = text.lower()
    # 4. Lemmatise
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Load the model and vectorizer
MODEL_FILE = 'svm_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    print(f"Error: Could not find {MODEL_FILE} or {VECTORIZER_FILE}.")
    print("Please make sure you have run the updated training notebook to save the models first!")
    exit(1)

print("Loading trained SVM model and TF-IDF vectorizer...")
svm_model = joblib.load(MODEL_FILE)
tfidf_vectorizer = joblib.load(VECTORIZER_FILE)
print("Models loaded successfully!\n")
print("="*60)
print("             MOVIE REVIEW SENTIMENT CLASSIFIER")
print("="*60)
print("Type a movie review and press Enter to get its sentiment.")
print("Type 'exit' or 'quit' to close the application.\n")

while True:
    try:
        user_input = input("\nEnter Review: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue
            
        # Preprocess
        cleaned_text = preprocess_text(user_input)
        
        # Vectorize
        features = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = svm_model.predict(features)[0]
        confidence = svm_model.decision_function(features)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        # Optional basic terminal text coloring
        color_code = '\033[92m' if prediction == 1 else '\033[91m'
        reset_code = '\033[0m'
        
        print(f"\n=> Sentiment: {color_code}{sentiment}{reset_code}")
        print(f"=> Decision Score: {confidence:.4f} ( > 0 is Positive, < 0 is Negative)")
        print("-" * 60)
                
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
