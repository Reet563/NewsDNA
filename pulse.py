import os
import feedparser
import json
import spacy
from gensim import corpora
from gensim.models import LdaMulticore

# --- 1. DYNAMIC PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# RSS Feeds to track (You can add more here!)
FEEDS = {

"TOI": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",

"Cricbuzz": "https://www.cricbuzz.com/rss/cricket-news",

"MoneyControl": "https://www.moneycontrol.com/rss/latestnews.xml",

"BBC Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml"}

class NewsPulse:
    def __init__(self):
        print("🚀 Initializing NewsDNA Pulse...")

        # Load the Model, Dictionary, and NLP
        try:
            self.model = LdaMulticore.load(os.path.join(MODEL_DIR, "lda_model.model"))
            self.dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary.dict"))
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

            # Load labels if they exist
            label_path = os.path.join(MODEL_DIR, "topic_labels.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.labels = json.load(f)
            else:
                self.labels = {}

            print("✅ Brain Loaded Successfully!")
        except Exception as e:
            print(f"❌ ERROR: Could not load model. Did you run train_v2.py first?")
            print(f"Details: {e}")
            exit()

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        return [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and t.is_alpha]

    def get_trends(self):
        # We use a dictionary to track the total "weight" of each topic found
        topic_totals = {str(i): 0.0 for i in range(self.model.num_topics)}
        article_count = 0

        print(f"📡 Scraping live feeds...")
        for source, url in FEEDS.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:  # Take top 15 from each source
                tokens = self.preprocess(entry.title)
                bow = self.dictionary.doc2bow(tokens)

                # Get topic distribution for this headline
                dist = self.model.get_document_topics(bow)
                for topic_id, prob in dist:
                    topic_totals[str(topic_id)] += prob
                article_count += 1

        # Calculate percentages
        results = []
        for t_id, total_score in topic_totals.items():
            percentage = (total_score / article_count) * 100

            # Get the label name or show top words if no label exists
            if t_id in self.labels:
                display_name = self.labels[t_id]
            else:
                # Show top 3 words of the topic so you know what it is
                top_words = ", ".join([word for word, prob in self.model.show_topic(int(t_id), 3)])
                display_name = f"Topic {t_id} ({top_words})"

            results.append((display_name, percentage))

        # Sort by highest percentage
        results.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "═" * 60)
        print(" 🔥  NewsDNA: LIVE TREND ANALYSIS  🔥 ")
        print("═" * 60)
        for name, score in results:
            bar_length = int(score / 2)  # 1 block per 2%
            bar = "█" * bar_length
            print(f"{name:<25} | {bar:<25} {round(score, 1)}%")
        print("═" * 60 + "\n")


if __name__ == "__main__":
    pulse = NewsPulse()
    pulse.get_trends()

    print("🔬 VERIFYING TOPIC DNA:")
    # Use the model's actual topic count instead of a hardcoded number
    for i in range(pulse.model.num_topics):
        words = [word for word, prob in pulse.model.show_topic(i, 5)]
        label = pulse.labels.get(str(i), "No Label")
        print(f"Topic {i} ({label}): {words}")
    print("═" * 60)