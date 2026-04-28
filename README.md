NewsDNA: Real-Time Sectoral Intelligence Pipeline

NewsDNA is an advanced news intelligence platform designed to decompose unstructured news headlines into a probabilistic "thematic DNA." Unlike traditional classifiers that force a story into a single category, NewsDNA utilizes Guided Latent Dirichlet Allocation (G-LDA) to reveal the nuanced mixture of topics within every article.

🚀 Key Features

Thematic DNA Extraction:
Analyzes news across 11 industrial and social sectors (Politics, Finance, Space, etc.)

Guided Intelligence:
Uses Semi-Supervised Priors to anchor AI clusters to real-world industrial categories

Linguistic Precision:
Powered by SpaCy for high-accuracy Noun-only filtering and lemmatization

Live Dashboard:
A Streamlit-based interface featuring real-time RSS scraping and interactive Plotly visualizations

Explainable AI (XAI):
Transparent probabilistic scoring instead of "black-box" classification

🏗️ The Architecture

Data Ingestion (pulse.py):
An automated ETL pipeline that scrapes live news via RSS feeds

Linguistic Processing:
SpaCy identifies and extracts the "thematic anchors" (Nouns and Proper Nouns)

Modeling Engine (train_v2.py):
A G-LDA model built with Gensim that uses custom Dirichlet Priors (η) to map text to specific sectors

UI Layer (app.py):
An interactive dashboard for Macro (Live Trends) and Micro (Deep URL) analysis

🛠️ Installation

Clone the repository:
git clone https://github.com/YOUR_USERNAME/NewsDNA.git
cd NewsDNA

Install dependencies:
pip install -r requirements.txt

Download the SpaCy model:
python -m spacy download en_core_web_sm

💻 Usage

1. Training the Model

To train the brain using your local dataset:
python train_v2.py --data master_trainer.csv --topics 11 --passes 20

2. Running the Dashboard

Launch the real-time monitoring tool:
streamlit run app.py

📊 Sectoral Categories

The model is specifically tuned for the Indian News Ecosystem, featuring categories such as:

Finance & Markets (Sensex, Nifty, Equity)
Space & Research (ISRO, Moon, Satellite)
Indian Politics (Sabha, Alliance, Ministry)
IT & Software (SaaS, Cloud, API)
...and 7 others

📂 Project Structure

app.py:
Streamlit UI and dashboard logic

train_v2.py:
The training pipeline and G-LDA implementation

pulse.py:
Live news scraping and data normalization

/model:
Contains saved LDA artifacts (model, dictionary, labels)

requirements.txt:
List of necessary Python libraries
