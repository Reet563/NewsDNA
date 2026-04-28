import streamlit as st
import pandas as pd
import os
import feedparser
import json
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
import plotly.express as px
from newspaper import Article

# --- 1. SETUP & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(page_title="NewsDNA Pulse", page_icon="🧬", layout="wide")
@st.cache_resource
def load_resources():
    model = LdaMulticore.load(os.path.join(MODEL_DIR, "lda_model.model"))
    dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary.dict"))
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    with open(os.path.join(MODEL_DIR, "topic_labels.json"), "r") as f:
        labels = json.load(f)
    return model, dictionary, nlp, labels

def preprocess(text, nlp):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and t.is_alpha]

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return f"{article.title} {article.text}"
    except Exception as e:
        return f"Error: Could not retrieve article. {e}"

st.title("🧬 NewsDNA: Intelligence Dashboard")

try:
    model, dictionary, nlp, labels = load_resources()
    tab1, tab2 = st.tabs(["📡 Live Trend Analysis", "🔬 Deep DNA Analysis"])

    with tab1:
        st.markdown("### Real-time Unsupervised Topic Modeling on Global News Feeds")
        FEEDS = {
            "TOI": "https://timesofindia.indiatimes.com/rssfeedmostrecent.cms",
            "Cricbuzz": "https://www.cricbuzz.com/rss/cricket-news",
            "MoneyControl": "https://www.moneycontrol.com/rss/business.xml",
            "BBC Tech": "https://feeds.bbci.co.uk/news/system/latest_published_content/rss.xml"
        }

        if st.button('📡 Refresh Live Analysis'):
            topic_totals = {str(i): 0.0 for i in range(model.num_topics)}
            article_count = 0
            all_data = []

            with st.spinner('Scraping live feeds...'):
                for source, url in FEEDS.items():
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:15]:
                        tokens = preprocess(entry.title, nlp)
                        bow = dictionary.doc2bow(tokens)
                        dist = model.get_document_topics(bow)

                        for t_id, prob in dist:
                            topic_totals[str(t_id)] += prob

                        main_topic_id = max(dist, key=lambda x: x[1])[0]
                        assigned_label = labels.get(str(main_topic_id), f"Topic {main_topic_id}")

                        all_data.append({
                            "Headline": entry.title,
                            "Source": source,
                            "Assigned DNA (Topic)": assigned_label
                        })
                        article_count += 1

            chart_list = []
            for t_id, total_score in topic_totals.items():
                perc = (total_score / article_count) * 100
                chart_list.append({"Category": labels.get(t_id), "Score": round(perc, 1)})
            df_chart = pd.DataFrame(chart_list).sort_values(by="Score", ascending=False)
            fig = px.bar(df_chart, x="Score", y="Category", orientation='h', title="Live Topic Dominance (%)",
                         color="Score")
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
            st.subheader("📰 Live Classified Headlines")
            final_table = pd.DataFrame(all_data)
            st.dataframe(
                final_table,
                column_config={
                    "Headline": st.column_config.TextColumn("News Headline", width="large"),
                    "Source": st.column_config.TextColumn("Source Feed", width="small"),
                    "Assigned DNA (Topic)": st.column_config.TextColumn("AI Classification", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
            st.success(f"Successfully processed {article_count} headlines.")

    with tab2:
        st.markdown("### Single Article DNA Analysis")
        st.write("Analyze the specific topic distribution of a single news link or custom text.")

        user_input = st.text_area("Paste a News URL or Article Text:", height=150,
                                  placeholder="https://... or 'Tata expands manufacturing in India...'")

        if st.button("🔬 Analyze DNA Percentages"):
            if user_input:
                with st.spinner("Extracting and Analyzing..."):
                    if user_input.startswith("http"):
                        raw_text = extract_article_text(user_input)
                    else:
                        raw_text = user_input

                    if "Error:" in raw_text:
                        st.error(raw_text)
                    else:
                        tokens = preprocess(raw_text, nlp)
                        bow = dictionary.doc2bow(tokens)
                        dist = model.get_document_topics(bow, minimum_probability=0.0)

                        dna_list = []
                        for t_id, prob in dist:
                            dna_list.append({
                                "Topic": labels.get(str(t_id), f"Topic {t_id}"),
                                "Match %": round(prob * 100, 2)
                            })

                        df_dna = pd.DataFrame(dna_list).sort_values(by="Match %", ascending=False)
                        top_match = df_dna.iloc[0]
                        st.metric("Primary Category", top_match['Topic'], f"{top_match['Match %']}% Match")

                        fig_dna = px.bar(df_dna, x="Match %", y="Topic", orientation='h',
                                         title="Article DNA Breakdown",
                                         color="Match %", color_continuous_scale="Viridis")
                        st.plotly_chart(fig_dna, use_container_width=True)

                        with st.expander("View Cleaned Text Used for Analysis"):
                            st.write(raw_text[:1000] + "...")
            else:
                st.warning("Please enter a link or text to analyze.")

except Exception as e:
    st.error(f"Error: {e}. Make sure you ran the training script first!")