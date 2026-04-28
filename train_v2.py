import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
import os
import json

import numpy as np


def get_guided_eta(dictionary, num_topics):
    eta = np.full((num_topics, len(dictionary)), 1.0)
    strong_dna = 2000.0
    weak_dna = 500.0
    seeds = {
        0: [
            'sabha', 'election', 'nda', 'gandhi', 'alliance', 'voter', 'bjp', 'congress',
            'modi', 'constituency', 'polling', 'manifesto', 'parliament', 'cabinet', 'ministry',
            'opposition', 'ballot', 'verdict', 'majority', 'pms', 'incumbent', 'legislature',
            'governance', 'democracy', 'exit', 'seat', 'sharing', 'strategy', 'campaign'
        ],  # Indian Politics
        1: [
            'bollywood', 'box', 'office', 'tollywood', 'trailer', 'star', 'cinema', 'movie',
            'theatre', 'film', 'actor', 'actress', 'director', 'release', 'screen', 'blockbuster',
            'sequel', 'premiere', 'production', 'entertainment', 'hero', 'villain', 'script',
            'teaser', 'Review'
        ],  # Cinema
        2: [
            'placement', 'iit', 'student', 'campus', 'exam', 'marks', 'syllabus', 'degree',
            'university', 'college', 'academic', 'semester', 'education', 'learning', 'faculty',
            'admission', 'course', 'curriculum', 'scholarship', 'result', 'cutoff', 'board',
            'tuition', 'study', 'graduate'
        ],  # Education
        3: [
            'sensex', 'nifty', 'market', 'trade', 'dividend', 'profit', 'stock', 'share',
            'equity', 'investor', 'bullish', 'bearish', 'index', 'portfolio', 'financial',
            'quarterly', 'revenue', 'earning', 'loss', 'valuation', 'capital', 'brokerage',
            'benchmark', 'trading', 'banking'
        ],  # Finance
        4: [
            'un', 'security', 'diplomacy', 'geopolitics', 'china', 'us', 'russia', 'israel',
            'ukraine', 'treaty', 'sanction', 'ambassador', 'global', 'international', 'border',
            'conflict', 'summit', 'bilateral', 'envoy', 'nato', 'peace', 'embassy',
            'foreign', 'territory',
        ],  # International Relations
        5: [
            'expert', 'analysis', 'view', 'opinion', 'column', 'doctor', 'insight', 'review',
            'forecast', 'impact', 'challenge', 'solution', 'health', 'condition', 'advice',
            'trend', 'research', 'finding', 'perspective', 'interview', 'debate',
            'report', 'professional', 'outlook'
        ],  # Expert Analysis
        6: [
            'match', 'ipl', 'bcci', 'fifa', 'cricket', 'stadium', 'tournament', 'player',
            'wicket', 'score', 'ball', 'over', 'world', 'cup', 'trophy', 'batting', 'bowling',
            'captain', 'umpire', 'inning', 'allrounder', 'final', 'semifinal', 'olympic', 'squad'
        ],  # Sports (Separate)
        7: [
            'isro', 'mission', 'moon', 'space', 'satellite', 'nasa', 'launch',
            'orbit', 'rocket', 'cosmos', 'exploration', 'astronomy', 'payload', 'scientist',
            'lunar', 'galaxy', 'telescope', 'propulsion', 'craft', 'physics', 'mars',
            'solar', 'earth', 'landing'
        ],  # Space & Research
        8: [
            'saas', 'software', 'platform', 'enterprise', 'cloud', 'application', 'developer',
            'startup', 'innovation', 'api', 'computing', 'tech', 'system', 'code', 'database',
            'backend', 'frontend', 'server', 'automation', 'digital', 'cyber',
            'encryption', 'ai', 'integration'
        ],
        9: [
            'arrested', 'killed', 'injured', 'death', 'accident', 'police', 'accused', 'court',
            'crime', 'murder', 'theft', 'fraud', 'collision', 'dead', 'hospital', 'investigation',
            'fir', 'jail', 'custody', 'victim', 'suspect', 'highway', 'rescue', 'emergency', 'guilty'
        ],
        10: [
            'air', 'india', 'tata', 'reliance', 'adani', 'mahindra', 'aviation', 'airline',
            'manufacturing', 'infrastructure', 'industry', 'corporate', 'ceo', 'acquisition',
            'merger', 'conglomerate', 'expansion', 'group', 'partnership', 'hcl', 'infosys',
            'transport', 'manufacturing', 'energy'
        ]
        # IT & Software (Separate)
    }
    for topic_id, words in seeds.items():
        for word in words:
            if word in dictionary.token2id:
                eta[topic_id, dictionary.token2id[word]] *= 1000
    return eta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess(text):

    junk_words = ["said", "added", "says", "told", "tuesday", "monday"]
    doc = nlp(str(text).lower())
    return [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"]
            and not t.is_stop and t.is_alpha and t.lemma_ not in junk_words]


def train(data_file, num_topics, iterations):

    print(f"📂 Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"❌ CSV not found: {data_file}")
        return

    print("🧠 Teaching the AI...")

    processed_docs = df['text'].map(preprocess)
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.2)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print(f"🧬 Generating DNA Anchors for {num_topics} topics...")
    eta_matrix = get_guided_eta(dictionary, num_topics)


    print(f"🏗️ Building Guided LDA Model with {num_topics} topics and {iterations} passes...")
    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=iterations,
        eta=eta_matrix,
        random_state=42,
        workers=3
    )

    lda.save(os.path.join(MODEL_DIR, "lda_model.model"))
    dictionary.save(os.path.join(MODEL_DIR, "dictionary.dict"))

    labels = {
        "0": "Indian Politics",
        "1": "Cinema & Box Office",
        "2": "Education & Campus",
        "3": "Finance & Markets",
        "4": "International Relations (UN)",
        "5": "Expert Analysis",
        "6": "Sports",
        "7": "Space & Research",
        "8": "IT & Software",
        "9": "CRIME & ACCIDENTS",
        "10": "INDUSTRIAL NEWS"
    }
    with open(os.path.join(MODEL_DIR, "topic_labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"✅ Training Complete. New brain saved in: {MODEL_DIR}")
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="news_data.csv", help="Name of the CSV file")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics")
    parser.add_argument("--passes", type=int, default=15, help="Training passes")
    args = parser.parse_args()


    train(data_file=args.data, num_topics=args.topics, iterations=args.passes)