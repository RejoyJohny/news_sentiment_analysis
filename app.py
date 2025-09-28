# app.py
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime
import glob



all_files = glob.glob("processed_news/*.csv")

if all_files:
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
else:
    df = pd.DataFrame(columns=["text", "publishedAt", "source", "category", "sent_label"])


# Keep your sidebar filters and visualization as-is

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data(ttl=300)
def fetch_news(api_key, page_size=50):
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize={page_size}&apiKey={api_key}"
    r = requests.get(url, timeout=10)
    articles = r.json().get("articles", [])
    rows = []
    for a in articles:
        text = (a.get("title") or "") + ". " + (a.get("description") or "")
        rows.append({
            "text": text,
            "publishedAt": a.get("publishedAt"),
            "source": a.get("source", {}).get("name"),
            "category": "general"
        })
    return pd.DataFrame(rows)
def save_news(news_df):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    news_df.to_csv(f"news_stream/news_{timestamp}.csv", index=False) 
def main():
    st.title("📰 News Sentiment Dashboard")
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        st.error("No NEWSAPI_KEY found. Add it in your app's Secrets on Streamlit Cloud.")
        st.stop()

    model = load_model()
    df = fetch_news(api_key)
    save_news(df)
    if df.empty:
        st.warning("No articles fetched.")
        return

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")


    st.sidebar.header("Filters")
    # Date filter
    min_date = df["publishedAt"].min().date()
    max_date = df["publishedAt"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
    if isinstance(date_range, tuple):
        df = df[(df["publishedAt"].dt.date >= date_range[0]) & (df["publishedAt"].dt.date <= date_range[1])]

    # Source filter
    sources = sorted(df["source"].dropna().unique().tolist())
    sel_sources = st.sidebar.multiselect("Source(s)", sources, default=sources)
    df = df[df["source"].isin(sel_sources)]

    st.markdown(f"### Showing {len(df)} news items")
    for _, row in df.iterrows():
        st.markdown(f"#### {row['text']}")
        st.write(f"📅 {row['publishedAt'].date() if pd.notna(row['publishedAt']) else 'Unknown'}  | 🗞️ {row['source']}  | {row['sent_label']}")
        st.markdown("---")

if __name__ == "__main__":
    main()
