# app.py
import json
import os
import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# --- BERTopic ---
from bertopic import BERTopic
from rank_bm25 import BM25Okapi

# Load .env
load_dotenv()
import os
os.environ['PYTORCH_JIT_LOG_LEVEL'] = '0'  # Suppress torch warning
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

CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    Ensure the DB schema supports topics and vectors.
    1. Add topic_id and embedding columns to umd_events if missing.
    2. Create topic_labels table if missing.
    3. Enable pgvector extension.
    4. Create HNSW index for fast vector search (if embeddings exist).
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 1. Add topic_id to events table if it doesn't exist
            cur.execute("""
                ALTER TABLE umd_events 
                ADD COLUMN IF NOT EXISTS topic_id INTEGER;
            """)
            
            # 2. Add embedding column (VECTOR(384) for MiniLM)
            cur.execute("""
                ALTER TABLE umd_events 
                ADD COLUMN IF NOT EXISTS embedding VECTOR(384);
            """)
            
            # 3. Create table for topic labels
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topic_labels (
                    topic_id INTEGER PRIMARY KEY,
                    label TEXT,
                    keywords TEXT
                );
            """)
            
            # 4. Create HNSW index for cosine similarity (fast ANN search)
            # Note: This assumes normalized embeddings; use vector_cosine_ops for exact cosine
            cur.execute("""
                CREATE INDEX IF NOT EXISTS umd_events_embedding_idx 
                ON umd_events USING hnsw (embedding vector_cosine_ops);
            """)
            
        conn.commit()
        print("✅ DB schema updated with pgvector support.")
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

import json  # Ensure this is imported at top of file

# ... existing imports ...

def fetch_events_with_embeddings() -> List[Tuple[dict, np.ndarray]]:
    """
    Fetch all events with their embeddings from DB.
    Fixes: Parses string representations of vectors if pgvector adapter isn't active.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT id, event, date, time, url, location, description, topic_id, embedding
                FROM umd_events
                ORDER BY id;
            """)
            rows = cur.fetchall()
        
        results = []
        zero_emb = np.zeros(384, dtype=np.float32) 
        
        for row in rows:
            ev = dict(row)
            emb_obj = row['embedding']
            
            # 1. Handle None
            if emb_obj is None:
                results.append((ev, zero_emb.copy()))
                continue
            
            # 2. Handle String (The fix for your Warnings)
            # Postgres returns '[0.1, ...]' string if pgvector isn't registered
            if isinstance(emb_obj, str):
                try:
                    # Postgres vector format looks like JSON array: [1,2,3]
                    parsed_emb = json.loads(emb_obj)
                    embedding = np.array(parsed_emb, dtype=np.float32)
                    results.append((ev, embedding))
                except Exception as e:
                    print(f"Error parsing string embedding for event {ev['id']}: {e}")
                    results.append((ev, zero_emb.copy()))
                continue
            
            # 3. Handle Bytes (if binary protocol used)
            if isinstance(emb_obj, (bytes, bytearray)):
                embedding = np.frombuffer(emb_obj, dtype=np.float32)
                results.append((ev, embedding))
                continue

            # 4. Handle List (if psycopg2 converted it already)
            if isinstance(emb_obj, list):
                embedding = np.array(emb_obj, dtype=np.float32)
                results.append((ev, embedding))
                continue

            # Fallback
            results.append((ev, zero_emb.copy()))
        
        return results
    finally:
        conn.close()

def build_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    Load events and embeddings from DB.
    Fixes: Safe normalization to prevent NaN crash.
    """
    event_emb_pairs = fetch_events_with_embeddings()
    if not event_emb_pairs:
        return [], np.empty((0, 384), dtype="float32")

    # Unzip
    events = []
    embeddings_list = []
    
    # Filter out pure zero vectors (invalid data) so they don't mess up Topic Modeling
    for ev, emb in event_emb_pairs:
        if np.count_nonzero(emb) > 0:
            events.append(ev)
            embeddings_list.append(emb)
            
    if not events:
        return [], np.empty((0, 384), dtype="float32")

    embeddings = np.stack(embeddings_list)  # Shape: (N, 384)

    # Safe Normalization (The fix for the Crash)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero: replace 0.0 norms with 1.0
    # (0 vector / 1.0 is still 0 vector, but it avoids NaN)
    norm[norm == 0] = 1.0 
    
    embeddings = embeddings / norm

    return events, embeddings

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
#  ML / EMBEDDING / INDEXES
# ============================

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


@st.cache_resource
def get_groq_client() -> Groq:
    """
    Get cached Groq client; minimal init to avoid proxies arg.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY env var missing. Check .env file.")
    
    # FIXED: Minimal init (no extra args; SDK 0.4.1+ handles proxies internally)
    return Groq(api_key=api_key)


def build_event_text(ev: dict) -> str:
    """
    Single representation used by:
      - dense embeddings
      - BM25 keyword search
      - Cross-Encoder re-ranking
    """
    parts = [
        ev.get("event") or "",
        f"Date: {ev.get('date')}",
        f"Time: {ev.get('time')}",
        ev.get("description") or "",
        ev.get("location") or ""
    ]
    return " ".join(p.strip() for p in parts if p)


@st.cache_data(show_spinner="Loading semantic (vector) index from DB...")  # Use cache_data for DB fetches (immutable)
def build_semantic_index() -> Tuple[List[dict], np.ndarray]:
    """
    Load events and embeddings from DB.
    Fixes: Safe normalization to prevent NaN crash.
    """
    event_emb_pairs = fetch_events_with_embeddings()
    if not event_emb_pairs:
        return [], np.empty((0, 384), dtype="float32")

    # Unzip
    events = []
    embeddings_list = []
    
    # Filter out pure zero vectors (invalid data) so they don't mess up Topic Modeling
    for ev, emb in event_emb_pairs:
        if np.count_nonzero(emb) > 0:
            events.append(ev)
            embeddings_list.append(emb)
            
    if not events:
        return [], np.empty((0, 384), dtype="float32")

    embeddings = np.stack(embeddings_list)  # Shape: (N, 384)

    # Safe Normalization (The fix for the Crash)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero: replace 0.0 norms with 1.0
    # (0 vector / 1.0 is still 0 vector, but it avoids NaN)
    norm[norm == 0] = 1.0 
    
    embeddings = embeddings / norm

    return events, embeddings

def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25: lowercase alphanumeric tokens.
    """
    return re.findall(r"\w+", (text or "").lower())


@st.cache_resource(show_spinner="Building BM25 keyword index...")
def build_bm25_index() -> Tuple[List[dict], Optional[BM25Okapi]]:
    """
    Build BM25 index over the same event texts used for embeddings.
    """
    events = fetch_all_events()
    if not events:
        return [], None

    texts = [build_event_text(ev) for ev in events]
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return events, bm25


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
        if tid != -1:  # Skip noise
            keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]
            label = generate_topic_label(keywords)
            structured_topics.append({
                "topic_id": int(tid),
                "keywords": keywords,
                "label": label
            })
            
    # 4. Save Labels to DB
    save_topic_labels(structured_topics)
    
    # Clear caches so next search picks up new topics & indexes
    build_semantic_index.clear()
    build_bm25_index.clear()
    
    return f"Successfully processed {len(structured_topics)} topics and updated database."


# ============================
#  HYBRID SEARCH (BM25 + Dense + RRF + Cross-Encoder)
# ============================

def semantic_search(
    query: str,
    top_k: int = 5,
    filter_topic_id: Optional[int] = None
) -> List[Tuple[dict, float]]:
    """
    Hybrid search with pgvector for dense retrieval + topic filtering.
    Fixed SQL WHERE clause construction for syntax correctness.
    """
    # --- Load BM25 index (unchanged) ---
    events_bm25, bm25 = build_bm25_index()
    if bm25 is None:
        return []

    N = len(events_bm25)
    
    # --- Dense retrieval with pgvector (FIXED: Robust WHERE/params) ---
    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype("float32")
    q_emb_list = q_emb.tolist()
    
    # Build conditions dynamically
    conditions = []
    params = []
    
    if filter_topic_id is not None:
        conditions.append("topic_id = %s")
        params.append(filter_topic_id)
    
    conditions.append("embedding IS NOT NULL")  # Always filter non-null
    
    where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""  # FIXED: Proper WHERE if any conds
    full_sql = f"""
        SELECT id, (embedding <=> %s::vector) AS dist
        FROM umd_events
        {where_sql}
        ORDER BY dist
        LIMIT 50;
    """
    
    # Params: Filter first, then always q_emb_list
    params.append(q_emb_list)
    
    # Log for debug (optional; remove in prod)
    print(f"Debug SQL: {full_sql}")  # Remove after test
    print(f"Debug params: {len(params)} values (filter: {filter_topic_id})")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(full_sql, params)
            dense_results = cur.fetchall()  # [(event_id, distance)]
    except Exception as e:
        print(f"Error in dense SQL: {e}. SQL: {full_sql}. Falling back to BM25-only.")
        dense_results = []  # Fallback to BM25
    finally:
        conn.close()
    
    # Map back to indices and convert dist to sim (unchanged)
    dense_indices = []
    for event_id, dist in dense_results:
        idx = next((i for i, ev in enumerate(events_bm25) if ev['id'] == event_id), None)
        if idx is not None:
            sim_score = 1.0 - dist  # L2 dist to sim (normalized)
            dense_indices.append((idx, sim_score))
    
    k_dense = min(50, len(dense_indices))
    dense_rank_indices = [idx for idx, _ in sorted(dense_indices, key=lambda x: x[1], reverse=True)][:k_dense] or np.array([])  # Empty fallback
    
    # --- BM25 retrieval (unchanged) ---
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    k_bm25 = min(50, N)
    bm25_rank_indices = np.argsort(-bm25_scores)[:k_bm25]

    # --- Reciprocal Rank Fusion (RRF) (unchanged) ---
    K = 60.0
    rrf_scores: Dict[int, float] = {}
    
    def add_rrf(indices: np.ndarray):
        for rank, idx in enumerate(indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (K + rank + 1.0)
    
    add_rrf(dense_rank_indices)
    add_rrf(bm25_rank_indices)
    
    if not rrf_scores:
        return []
    
    fused_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    # --- Cross-Encoder re-ranking (unchanged) ---
    cross_encoder = get_cross_encoder()
    events = events_bm25
    docs_texts = [build_event_text(ev) for ev in events]

    candidate_pairs = [(query, docs_texts[idx]) for idx in fused_indices]
    ce_scores = cross_encoder.predict(candidate_pairs)

    ranked: List[Tuple[dict, float]] = []
    for idx, score in sorted(
        zip(fused_indices, ce_scores),
        key=lambda x: x[1],
        reverse=True
    ):
        ev = events[idx]

        # Topic filter (post-fusion)
        if filter_topic_id is not None and ev.get("topic_id") != filter_topic_id:
            continue

        ranked.append((ev, float(score)))
        if len(ranked) >= top_k:
            break

    return ranked


# ============================
#  RAG ANSWERING
# ============================

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
    current_date = datetime.now().strftime("%B %Y")

    sys_prompt = """
    You are TestudoBot, a knowledgeable, friendly, and enthusiastic AI assistant for University of Maryland (UMD) events. Your goal: Help users discover lectures, career fairs, performances, workshops, and more – using ONLY the provided Context. Do not invent details, dates, or URLs. If Context lacks info, say so politely and suggest alternatives from data.

    Key Parameters:
    - Current Date: {current_date} (resolve relatives like "tomorrow" or "next week" based on this).
    - Known Categories: {topic_list} (map user terms closely, e.g., "career" → "Career Services").
    - Max Results: {max_results} (prioritize top matches).

    Instructions:
    1. Think step-by-step internally: (a) Parse query for elements (date/time relative/absolute, category/topic, location, audience, constraints like "virtual" or "free"). (b) Match to Context: Prioritize upcoming events (filter past unless asked; flex ±3 days for relatives). Deduplicate; rank by soonest date, category match, then search score. Diversify broad queries across categories. (c) Summarize 1-{max_results} best fits.
    2. No exact matches? "I couldn't find matching events in the current data." Suggest 1-2 closest (e.g., nearby dates/topics) or ask: "What details can you add (date/type)?"
    3. Maintain conversation: Reference history for preferences (e.g., "Building on your interest in evenings...").

    Output Rules:
    - Concise (<180 words), engaging, and polite academic tone.
    - Structure: Direct answer sentence (note date interpretation if key). Bullet list of top events:
    - **Event Name**
    - Date/Time
    - Location (if available)
    - Brief Description (≤20 words)
    - [Source: Event ID (URL if provided)]
    - Omit missing fields. End with a short follow-up if ambiguous (e.g., "What else interests you?").

    Examples:
    - Query: "Career events next week?" → "Upcoming Career Services events next week: \n• **Job Fair** \nDate/Time: Oct 10, 2-5pm \nLocation: Stamp Student Union \nDesc: Networking for students and grads. \n[Source: Event 123 (https://umd.edu/fair)]"
    - Query: "Virtual events tomorrow?" (None) → "No virtual events tomorrow, but here's a close in-person alternative on Oct 9. Interested in more options?"
    - Query: "Fun events this month?" (Broad) → "Diverse fun events this month: \n• **Concert Series** \nDate/Time: Oct 15, 7pm \nLocation: Clarice Smith Center \nDesc: Live music performances. \n[Source: Event 456] \nPrefer a specific genre?"

    User Query (with history): {query}
    Context: {context_text}"""
     
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

@st.cache_resource
def run_startup_sequence():
    """
    Runs once when the server starts. 
    Checks if topics exist. If not, runs the pipeline automatically.
    Returns: True if updates were made (requiring a rerun), False otherwise.
    """
    print("🚀 Checking system status...")
    
    # Check if we already have topics in the DB
    existing_topics = fetch_topic_map()
    
    if not existing_topics:
        print("⚠️ No topics found. Auto-starting Topic Modeling Pipeline...")
        
        # Run the pipeline (this function ALREADY clears the caches)
        msg = run_pipeline_and_save_topics()
        
        print(f"✅ Startup Sequence Complete: {msg}")
        return True # <--- RETURN TRUE (Update happened)
    else:
        print(f"✅ System ready. Found {len(existing_topics)} existing topics.")
        return False # <--- RETURN FALSE (No update needed)
# ============================
#  UI MAIN
# ============================
 
# 1. Initialize DB schema on first run
init_db()

st.set_page_config(page_title="UMD Smart Events", page_icon="🐢", layout="wide")
st.title("🐢 UMD Events RAG + Topic Modeling")

# --- NEW: Run the startup check immediately ---
with st.spinner("Initializing Application and AI Models..."):
    # This runs the topic model ONLY if DB is empty of topics
    # @st.cache_resource ensures it happens once per server restart
    run_startup_sequence()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch topics for Sidebar (Now guaranteed to exist!)
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
    
    # Topic modeling pipeline trigger
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
        # 1. Hybrid Search with Filter
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
                        st.caption(
                            f"Topic: {available_topics.get(ev.get('topic_id'), 'Uncategorized')}"
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
