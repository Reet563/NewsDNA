NewsDNA: Real-Time News Intelligence Dashboard
Unsupervised Topic Modeling with Semi-Supervised DNA Anchors
1. Project Overview

NewsDNA is a news classification and intelligence engine designed to analyze the “DNA” of the Indian news ecosystem.

Unlike traditional classifiers that assign a single label, NewsDNA treats every article as a mixture of 11 industrial and social segments, providing a percentage-based thematic breakdown.

The system uses Guided Latent Dirichlet Allocation (G-LDA) to combine:

Unsupervised topic discovery
Human-defined domain knowledge
2. Core Features
📡 Live Trend Analysis
Scrapes real-time RSS feeds (TOI, MoneyControl, Cricbuzz)
Visualizes the current “Macro Pulse” of media trends
🔬 Deep DNA Analysis
Uses newspaper3k to extract article content from URLs
Generates a probability distribution of topics
🧬 Guided Inference
Uses Seed Anchors for domain-aware modeling
Example: Sabha, Sensex, IPL → improves Indian context accuracy
🛡️ Confidence Thresholding
Filters out low-confidence predictions
Labels uncertain outputs as “General News”
3. System Architecture

The system is divided into two main phases:

Phase 1: Knowledge Acquisition (The Brain)

Data Ingestion

Loads master_trainer.csv for base vocabulary

NLP Pipeline

Uses SpaCy for POS filtering
Keeps only:
Nouns
Proper Nouns
Reduces semantic noise

Model Training

Trains a Gensim LDA model
Uses a custom η (Eta) Priority Matrix
Injects seed word bias (Guided LDA)
Phase 2: Inference Dashboard (The Interface)

Frontend

Built with Streamlit
Tab-based responsive UI

Live Data

Uses feedparser for real-time RSS feeds

Visualization

Uses Plotly Express
Displays interactive horizontal bar charts
4. Tech Stack
Component	Technology
Language	Python 3.x
Modeling	Gensim (Guided LDA)
NLP	SpaCy (en_core_web_sm)
Frontend	Streamlit
Data Mining	Feedparser, Newspaper3k
Visualization	Plotly, Pandas
5. Installation & Usage
Prerequisites
pip install streamlit gensim spacy pandas plotly feedparser newspaper3k
python -m spacy download en_core_web_sm
Training the Model
python train_v2.py --data master_trainer.csv --topics 11 --passes 50
Running the Dashboard
streamlit run app.py
6. Technical Challenges Overcome
Topic Gravity
Problem: Words like “Indian” biased topics (e.g., Sports/Politics)
Solution: Implemented 40% Confidence Threshold
Semantic Drift
Problem: Model focused on writing style instead of meaning
Solution: Applied POS Tagging (Nouns only)
Dual-Stream Ingestion
Problem: Static training vs real-time data mismatch
Solution: Integrated:
Static dataset (training)
Live RSS feeds (inference)
7. Project DNA Categories
Indian Politics
Cinema & Box Office
Education & Campus
Finance & Markets
International Relations
Expert Analysis
Sports
Space & Research
IT & Software
Crime & Legal
Industrial News