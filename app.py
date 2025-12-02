# app.py

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# --- BERTopic ---
from bertopic import BERTopic

# Load .env
load_dotenv()

# ============================
#  CONFIG
# ============================

DB_NAME = os.getenv("DB_NAME", "umd_events")
DB_USER = os.getenv("DB_USER", "umd_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "umd_password")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================
#  DATABASE HELPERS
# ============================

def get_db_connection():
    """Create a new PostgreSQL connection."""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    return conn

def init_db():
    """
    Ensure the DB schema supports topics.
    1. Add topic_id column to umd_events if missing.
    2. Create topic_labels table if missing.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Add topic_id to events table if it doesn't exist
            cur.execute("""
                ALTER TABLE umd_events 
                ADD COLUMN IF NOT EXISTS topic_id INTEGER;
            """)
            
            # 2. Create table for topic labels
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topic_labels (
                    topic_id INTEGER PRIMARY KEY,
                    label TEXT,
                    keywords TEXT
                );
            """)
        conn.commit()
    except Exception as e:
        print(f"DB Init Error: {e}")
        conn.rollback()
    finally:
        conn.close()

def fetch_all_events() -> List[dict]:
    """Load ALL events, including their assigned topic_id."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT id, event, date, time, url, location, description, topic_id
                FROM umd_events
                ORDER BY id;
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def fetch_topic_map() -> Dict[int, str]:
    """Returns a dict mapping topic_id -> Label (e.g. {0: 'Career Fair', 1: 'Music'})"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT topic_id, label FROM topic_labels WHERE topic_id != -1 ORDER BY label;")
            rows = cur.fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}
    finally:
        conn.close()

def update_event_topics(event_ids: List[int], topic_ids: List[int]):
    """Bulk update topic_ids in the main events table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for eid, tid in zip(event_ids, topic_ids):
                cur.execute(
                    "UPDATE umd_events SET topic_id = %s WHERE id = %s;",
                    (int(tid), int(eid))
                )
        conn.commit()
    finally:
        conn.close()

def save_topic_labels(topic_data: List[dict]):
    """Save the AI-generated labels to the topic_labels table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Clear old labels first
            cur.execute("TRUNCATE TABLE topic_labels;")
            
            for t in topic_data:
                cur.execute(
                    """
                    INSERT INTO topic_labels (topic_id, label, keywords)
                    VALUES (%s, %s, %s);
                    """,
                    (t['topic_id'], t['label'], ", ".join(t['keywords']))
                )
        conn.commit()
    finally:
        conn.close()


# ============================
#  ML / EMBEDDING
# ============================

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)

def build_event_text(ev: dict) -> str:
    parts = [
        ev.get("event") or "",
        f"Date: {ev.get('date')}",
        f"Time: {ev.get('time')}",
        ev.get("description") or "",
        ev.get("location") or ""
    ]
    return " ".join(p.strip() for p in parts if p)

@st.cache_resource(show_spinner="Building semantic index...")
def build_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    Fetch events and build/cache embeddings.
    """
    events = fetch_all_events()
    if not events:
        return [], np.empty((0, 384), dtype="float32")

    model = get_embedding_model()
    texts = [build_event_text(ev) for ev in events]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    return events, embeddings


# ============================
#  TOPIC MODELING & LLM
# ============================

def generate_topic_label(keywords: List[str]) -> str:
    """Asks Groq LLM to name a cluster."""
    client = get_groq_client()
    prompt = f"""
    Keywords: {', '.join(keywords)}.
    Provide a category name (max 4 words) for this event cluster. 
    Examples: "Career Services", "Arts & Performance".
    Return ONLY the name.
    """
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15, temperature=0.3
        )
        return resp.choices[0].message.content.strip().replace('"', '')
    except:
        return "General Events"

def run_pipeline_and_save_topics():
    """
    1. Load data & embeddings.
    2. Run BERTopic.
    3. Generate Labels via LLM.
    4. Save EVERYTHING to Postgres.
    """
    events, embeddings = build_semantic_index()
    if len(events) < 5:
        return "Not enough data to model topics."

    docs = [build_event_text(ev) for ev in events]

    # 1. Run BERTopic
    topic_model = BERTopic(min_topic_size=3, verbose=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # 2. Update Events Table with Topic IDs
    event_ids = [ev['id'] for ev in events]
    update_event_topics(event_ids, topics)

    # 3. Generate Labels for discovered topics
    topic_info = topic_model.get_topic_info()
    structured_topics = []
    
    for index, row in topic_info.iterrows():
        tid = row['Topic']
        if tid != -1: # Skip noise
            keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]
            label = generate_topic_label(keywords)
            structured_topics.append({
                "topic_id": int(tid),
                "keywords": keywords,
                "label": label
            })
            
    # 4. Save Labels to DB
    save_topic_labels(structured_topics)
    
    # Clear cache so next search picks up new topics
    build_semantic_index.clear()
    
    return f"Successfully processed {len(structured_topics)} topics and updated database."


# ============================
#  SEARCH & RAG
# ============================

def semantic_search(query: str, top_k: int = 5, filter_topic_id: Optional[int] = None) -> List[Tuple[dict, float]]:
    """
    Search with optional Topic Filtering.
    """
    events, emb_matrix = build_semantic_index()
    if not events: 
        return []

    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype("float32")

    scores = emb_matrix @ q_emb
    
    scored_events = []
    for idx, ev in enumerate(events):
        scored_events.append((ev, float(scores[idx])))

    # Apply Filtering if requested
    if filter_topic_id is not None:
        scored_events = [
            (ev, s) for (ev, s) in scored_events 
            if ev.get('topic_id') == filter_topic_id
        ]

    scored_events.sort(key=lambda x: x[1], reverse=True)

    return scored_events[:top_k]

def call_groq_rag(query, retrieved_events, history):
    client = get_groq_client()
    
    context_text = "\n\n".join([
        f"Event: {ev['event']}\nDate: {ev['date']}\nDesc: {ev['description'][:300]}"
        for ev, score in retrieved_events
    ])
    
    if not context_text:
        context_text = "No specific events found matching criteria."

    # Fetch topics to give the LLM context about what's available
    topic_map = fetch_topic_map()
    topic_list = ", ".join(topic_map.values())

    sys_prompt = (
        "You are a helpful UMD Events assistant. "
        f"Today's date is roughly Oct 1, 2025 (simulated). "
        f"The available event categories are: {topic_list}. "
        "Answer the user's question using the Context below. "
        "If the user asks about a specific topic, check if it matches one of the categories. "
        "If the context doesn't help, say so politely."
    )
    
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(history[-6:])
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context_text}\n\nUser Question: {query}"
    })

    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2
    )
    return resp.choices[0].message.content


# ============================
#  UI MAIN
# ============================

# 1. Initialize DB schema on first run
init_db()

# FIXED: Removed corrupted characters in page config
st.set_page_config(page_title="UMD Smart Events", page_icon="🐢", layout="wide")

# FIXED: Replaced mojibake with Turtle emoji
st.title("🐢 UMD Events RAG + Topic Modeling")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch topics for Sidebar
available_topics = fetch_topic_map() 
# {id: "Label"}

with st.sidebar:
    st.header("Search Filters")
    
    # TOPIC FILTER DROPDOWN
    selected_topic_label = st.selectbox(
        "Filter by Topic",
        options=["All Topics"] + list(available_topics.values())
    )
    
    # Map label back to ID
    filter_id = None
    if selected_topic_label != "All Topics":
        # Invert dict to find ID
        for tid, label in available_topics.items():
            if label == selected_topic_label:
                filter_id = tid
                break
    
    top_k = st.slider("Results (Top K)", 1, 10, 5)
    
    st.divider()
    st.header("Admin Controls")
    
    # FIXED: Replaced mojibake with Brain emoji
    if st.button("🧠 Analyze & Save Topics"):
        with st.spinner("Clustering events, generating labels, and saving to DB..."):
            msg = run_pipeline_and_save_topics()
            st.success(msg)
            # Force reload to update sidebar
            st.rerun()

# Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about events..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Searching..."):
        # 1. Search with Filter
        results = semantic_search(prompt, top_k=top_k, filter_topic_id=filter_id)
        
        # 2. Generate Answer
        answer = call_groq_rag(prompt, results, st.session_state.messages)
        
        # 3. Display
        with st.chat_message("assistant"):
            st.markdown(answer)
            if results:
                with st.expander("See Sources"):
                    for ev, score in results:
                        st.markdown(f"**{ev['event']}** ({score:.2f})")
                        st.caption(f"Topic: {available_topics.get(ev.get('topic_id'), 'Uncategorized')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})