# app.py
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime
import glob




all_files = glob.glob("processed_news/*.csv")

dfs = []
for f in all_files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"⚠️ Skipping {f} due to error: {e}")

if dfs:
    df = pd.concat(dfs, ignore_index=True)
else:
    df = pd.DataFrame(columns=["text", "publishedAt", "source", "category", "sentiment"])

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
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    news_df.to_csv(f"news_stream/news_{timestamp}.csv", index=False) 


def main():
    st.title("📰 News Sentiment Dashboard")
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        st.error("No NEWSAPI_KEY found. Add it in your app's Secrets on Streamlit Cloud.")
        st.stop()

    model = load_model()
    df = fetch_news(api_key)

    if df.empty:
        st.warning("No articles fetched.")
        return

    # Predict sentiment using your trained pipeline
    df["sentiment"] = model.predict(df["text"].fillna(""))
    df["sent_label"] = df["sentiment"].map({1: "✅ Positive", 0: "❌ Negative"})

    # Save news for streaming / record-keeping
    save_news(df)

    # Convert publishedAt to datetime
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    sent_label_col = "sent_label" if "sent_label" in df.columns else None

    # ---------------------------
    # Sidebar Filters
    # ---------------------------
    st.sidebar.header("Filters")
    min_date = df["publishedAt"].min().date()
    max_date = df["publishedAt"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
    if isinstance(date_range, tuple):
        df = df[(df["publishedAt"].dt.date >= date_range[0]) & (df["publishedAt"].dt.date <= date_range[1])]

    sources = sorted(df["source"].dropna().unique().tolist())
    sel_sources = st.sidebar.multiselect("Source(s)", sources, default=sources)
    df = df[df["source"].isin(sel_sources)]

    # ---------------------------
    # Display news
    # ---------------------------
    st.markdown(f"### Showing {len(df)} news items")
    for _, row in df.iterrows():
        date_str = row["publishedAt"].date() if pd.notna(row["publishedAt"]) else "Unknown"
        source_str = row["source"] if "source" in row else "Unknown"
        sent_str = row[sent_label_col] if sent_label_col else "❌ Unknown"

        st.markdown(f"#### {row['text']}")
        st.write(f"📅 {date_str}  | 🗞️ {source_str}  | {sent_str}")
        st.markdown("---")


if __name__ == "__main__":
    main()
        

