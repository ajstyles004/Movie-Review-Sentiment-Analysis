# Sentiment Classification of Movie Reviews Using Support Vector Machine 🎬

## Overview
This repository contains an end-to-end sentiment analysis project that classifies movie reviews from the IMDB Large Movie Review Dataset as positive or negative. The project implements a robust classical machine learning pipeline using **Term Frequency-Inverse Document Frequency (TF-IDF)** and a **Linear Support Vector Machine (LinearSVC)**, achieving strong predictive accuracy on high-dimensional text data.

This project was developed rigorously, exploring advanced natural language processing (NLP) preprocessing, model validation strategies, and a multi-metric statistical evaluation across varying train-test split conditions (80-20, 60-40, 50-50).

## Highlights and Features
* **Extensive Preprocessing:** HTML tag removal via BeautifulSoup, regex-driven noise cleaning, and WordNet-based lemmatisation to ensure a clean, linguistically valid tokenized corpus.
* **Optimized Feature Vectorization:** Features sublinear TF scaling (`1 + log(tf)`), bigram generation (`ngram_range=(1,2)`), and aggressive corpus-level frequency bounding to prevent vocabulary dominance and systematically capture semantic negations.
* **Rigorous Multi-Metric Evaluation:** Examines model performance far beyond simple accuracy tracking by employing Precision, Recall, F1-Score, AUC-ROC, Matthews Correlation Coefficient (MCC), and Cohen’s Kappa.
* **Visual Data Insights:** Generates visually impactful analyses including polarity word clouds, multi-model comparative bar plots, confusion matrices, ROC curve overlays, and an analysis of the top-weighted SVM decision coefficients.

## Project Structure
- `Create_Notebook.py`: Python script used to programmatically assemble and generate the original Jupyter Notebook.
- `Sentiment_Analysis_SVM.ipynb`: The primary executable environment; an end-to-end technical walkthrough with code, markup, and results.
- `Sentiment_Analysis_SVM.html`: An exported, read-only HTML report of the executed notebook for browser-based offline viewing.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ajstyles004/Movie-Review-Sentiment-Analysis.git
   cd Movie-Review-Sentiment-Analysis
   ```

2. **Initialize virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Mac/Linux
   pip install -r requirements.txt
   ```

3. **Dataset Acquisition:**
   The `aclImdb` dataset (Maas et al., 2011) consists of 50,000 highly polar movie reviews. Due to size limitations, the dataset is excluded (`.gitignore`) from source control. Download it manually from [Stanford's dataset repository](https://ai.stanford.edu/~amaas/data/sentiment/) or via standard loaders and extract it to the project root directory in a folder named `aclImdb`.

4. **Running the Analysis:**
   Open and execute `Sentiment_Analysis_SVM.ipynb` from top to bottom. The notebook natively handles automatic NLTK resource fetching (`wordnet`, `omw-1.4`) avoiding manual dictionary downloads.

## Key Findings & Results
Testing split ratios demonstrated that maximizing training data volume was the most dominant factor in improving model generalisation. The **80-20 split** robustly outperformed smaller dataset allocations:

- **Accuracy:** `~89.6%`
- **AUC-ROC:** `0.961`
- **F1-Score:** `0.897`

Extracting mathematical coefficients from the `LinearSVC` decision boundary successfully highlighted top positive identifiers (e.g. *"excellent"*, *"perfect"*, *"highly recommend"*) vs negative identifiers (e.g. *"worst"*, *"awful"*, *"waste time"*), proving the model learned semantically meaningful and compositional boundaries rather than arbitrary distributions.

## References
* **Dataset:** Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)*.
* **Theory:** Modeling patterns guided by established Information Retrieval and text categorization fundamentals as described by Joachims (1998) and Wang & Manning (2012).
