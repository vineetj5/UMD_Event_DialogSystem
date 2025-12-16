# app.py
import os
import json
import time
import logging
import chainlit as cl
from datetime import datetime
from typing import List, Dict
import numpy as np 
import psycopg2
from sentence_transformers import SentenceTransformer
from psycopg2 import pool
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from dateutil import parser  # ADD: pip install python-dateutil (for date parsing)
from dateutil.relativedelta import relativedelta

from datetime import datetime
from dateutil.relativedelta import relativedelta # Ensure you installed python-dateutil

import re
from dateutil import parser
from datetime import timedelta
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from contextlib import contextmanager

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
os.environ['PYTORCH_JIT_LOG_LEVEL'] = '0'

DB_NAME = os.getenv("DB_NAME", "umd_events")
DB_USER = os.getenv("DB_USER", "umd_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "umd_password")
DB_HOST = os.getenv("DB_HOST", "db")
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# 1. Model for simple tasks (Topic Labeling) - Faster & Cheaper
LABEL_MODEL = "meta-llama/llama-3.1-8b-instruct"

# 2. Model for complex tasks (Chat & RAG) - Smarter & More Detailed
CHAT_MODEL = "meta-llama/llama-3.1-70b-instruct"
SITE_URL = os.getenv("OR_SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("OR_SITE_NAME", "TestudoBot")

# --- GLOBAL CLIENTS ---
es_client = Elasticsearch(ELASTIC_HOST)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
)

# ============================
#  DATABASE & SEARCH LOGIC
# ============================

# 1. Create a Global Connection Pool
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432"
    )
    if db_pool:
        logger.info("✅ Database connection pool created successfully")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")

# 2. Helper to get a cursor
@contextmanager
def get_db_cursor(cursor_factory=None):
    conn = db_pool.getconn()
    try:
        if cursor_factory:
            yield conn.cursor(cursor_factory=cursor_factory)
        else:
            yield conn.cursor()
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        db_pool.putconn(conn)

def init_db():
    """Initializes the database schema."""
    try:
        with get_db_cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS topic_id INTEGER DEFAULT -1;")
            cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS embedding VECTOR(384);")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topic_labels (
                    topic_id INTEGER PRIMARY KEY,
                    label TEXT,
                    keywords TEXT
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS umd_events_embedding_idx 
                ON umd_events USING hnsw (embedding vector_cosine_ops);
            """)
            logger.info("✅ Database initialized.")
    except Exception as e:
        logger.error(f"❌ DB Init Error: {e}")
        # removed unsafe conn.rollback() here

def fetch_topic_map() -> Dict[str, str]:
    """Returns { 'Career': '1', 'Music': '2' } for the dropdown."""
    try:
        with get_db_cursor() as cur:
            # Check if table exists and has data
            cur.execute("SELECT COUNT(*) FROM topic_labels;")
            count = cur.fetchone()[0]
            logger.info(f"🔍 DEBUG: 'topic_labels' table contains {count} rows.")

            cur.execute("SELECT topic_id, label FROM topic_labels WHERE topic_id != -1 ORDER BY label;")
            rows = cur.fetchall()
            
            result = {row[1]: str(row[0]) for row in rows}
            logger.info(f"🔍 DEBUG: Topic Map returning: {result}")
            return result
    except Exception as e:
        logger.error(f"❌ ERROR in fetch_topic_map: {e}")
        return {}




# ------------------------------------------------------------------
# ROBUST HYBRID SEARCH  (drop-in replacement for search_events)
# ------------------------------------------------------------------
 

import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def search_events(
    query_text: str,
    top_k: int = 15,
    filter_topic_id: int | None = None,
) -> list[tuple[dict, float]]:
    """
    ULTIMATE HYBRID SEARCH (Fixed for 'Upcoming' Events):
    1. Parsing: Handles 'upcoming', 'next week', 'this month', and specific dates.
    2. Retrieval: Runs Vector (Semantic) and Keyword (BM25) searches.
    3. Fusion: Uses RRF for robust ranking.
    """
    current = datetime.now()
    q_lower = query_text.lower()
    cleaned_query = query_text
    start_date, end_date = None, None

    # --- 1. ROBUST DATE PARSING ---
    
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    # A. Specific Date: "November 13"
    specific_date_match = re.search(r'\b(' + '|'.join(months.keys()) + r')\s+(\d{1,2})\b', q_lower)
    
    if specific_date_match:
        m_name = specific_date_match.group(1)
        day_num = int(specific_date_match.group(2))
        m_num = months[m_name]
        year = current.year 
        try:
            target_date = datetime(year, m_num, day_num)
            start_date = target_date.strftime("%Y-%m-%d")
            end_date = target_date.strftime("%Y-%m-%d") 
            cleaned_query = re.sub(rf"\b{m_name}\s+{day_num}\b", "", cleaned_query, flags=re.I)
        except ValueError:
            logger.warning(f"Invalid date parsed: {m_name} {day_num}")

    # B. "Upcoming" / "Future" (The Fix!)
    # Sets filter from TODAY -> 1 Year from now
    elif "upcoming" in q_lower or "future" in q_lower or "soon" in q_lower:
        start_date = current.strftime("%Y-%m-%d")
        end_date = (current + relativedelta(years=1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\b(upcoming|future|soon)\b", "", cleaned_query, flags=re.I)

    # C. Relative Dates
    elif "this month" in q_lower:
        start_date = current.replace(day=1).strftime("%Y-%m-%d")
        end_date = (current + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bthis month\b", "", cleaned_query, flags=re.I)
    elif "next month" in q_lower:
        start = (current + relativedelta(months=1)).replace(day=1)
        start_date = start.strftime("%Y-%m-%d")
        end_date = (start + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bnext month\b", "", cleaned_query, flags=re.I)
    elif "next week" in q_lower:
        start_date = current.strftime("%Y-%m-%d")
        end_date = (current + timedelta(days=7)).strftime("%Y-%m-%d")
        cleaned_query = re.sub(r"\bnext week\b", "", cleaned_query, flags=re.I)

    # D. Whole Month: "In October"
    else:
        for m_name, m_num in months.items():
            if m_name in q_lower:
                year = current.year
                dt = datetime(year, m_num, 1)
                start_date = dt.strftime("%Y-%m-%d")
                end_date = (dt + relativedelta(months=1, days=-1)).strftime("%Y-%m-%d")
                cleaned_query = re.sub(rf"\b{m_name}\b", "", cleaned_query, flags=re.I)
                break

    # Strip filler words
    cleaned_query = re.sub(r"\b(events?|happening|show me|find|list|from|on|in)\b", "", cleaned_query, flags=re.I).strip()
    
    # If query became empty (e.g. "upcoming events" -> ""), reset it to find *anything* in range
    if not cleaned_query: 
        cleaned_query = query_text 

    # --- 2. BUILD FILTERS ---
    es_filters = []
    if filter_topic_id is not None:
        es_filters.append({"term": {"topic_id": filter_topic_id}})
    if start_date and end_date:
        es_filters.append({"range": {"date": {"gte": start_date, "lte": end_date, "format": "yyyy-MM-dd"}}})
    
    filter_query = {"bool": {"filter": es_filters}} if es_filters else None

    # --- 3. VECTOR SEARCH ---
    try:
        vector_body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": embedding_model.encode(cleaned_query, normalize_embeddings=True).tolist(),
                "k": top_k,
                "num_candidates": 200,
            },
            "_source": ["event", "description", "date", "time", "location", "url"]
        }
        if filter_query: vector_body["knn"]["filter"] = filter_query
        v_res = es_client.search(index="umd_events", body=vector_body)["hits"]["hits"]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        v_res = []

    # --- 4. KEYWORD SEARCH ---
    try:
        keyword_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": cleaned_query,
                            "fields": ["event^3", "description^2", "location"],
                            "fuzziness": "AUTO",
                            "operator": "or"
                        }
                    },
                    "filter": es_filters
                }
            },
            "_source": ["event", "description", "date", "time", "location", "url"]
        }
        k_res = es_client.search(index="umd_events", body=keyword_body)["hits"]["hits"]
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        k_res = []

    # --- 5. RRF FUSION ---
    def rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    fused = {} 
    for r, hit in enumerate(v_res):
        _id = hit["_id"]
        if _id not in fused: fused[_id] = {"doc": hit["_source"], "score": 0.0}
        fused[_id]["score"] += rrf_score(r)

    for r, hit in enumerate(k_res):
        _id = hit["_id"]
        if _id not in fused: fused[_id] = {"doc": hit["_source"], "score": 0.0}
        fused[_id]["score"] += rrf_score(r)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [(item["doc"], item["score"]) for item in ranked]
# --- PIPELINE & ADMIN TASKS ---

def run_pipeline_if_needed():
    """Checks if we need to run the categorization pipeline."""
    # Check for uncategorized events
    try:
        with get_db_cursor() as cur:
            # Count events that are either NULL or -1 (uncategorized)
            cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id IS NULL OR topic_id = -1;")
            uncategorized_count = cur.fetchone()[0]
    except Exception:
        uncategorized_count = 0

    topic_map = fetch_topic_map()
    
    # Run if topics are empty OR we have new uncategorized data
    if not topic_map or uncategorized_count > 0:
        logger.info(f"⚠️ Found {uncategorized_count} uncategorized events or no topics. Running Pipeline...")
        return run_topic_modeling_pipeline()
        
    logger.info("✅ Topics exist and data is categorized. Skipping pipeline.")
    return False

def run_topic_modeling_pipeline():
    """
    Runs BERTopic -> Updates Postgres -> Syncs Elasticsearch -> Generates Labels
    """
    from bertopic import BERTopic
    from psycopg2.extras import DictCursor

    logger.info("🚀 Starting Topic Modeling Pipeline...")

    # 1. Fetch Data
    events = []
    embeddings = []
    
    with get_db_cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT id, event, description, embedding FROM umd_events")
        rows = cur.fetchall()
        for r in rows:
            if r['embedding'] is None: continue
            emb = np.array(json.loads(r['embedding']) if isinstance(r['embedding'], str) else r['embedding'])
            if np.count_nonzero(emb) == 0: continue
            events.append(dict(r))
            embeddings.append(emb)

    if len(events) < 5: 
        logger.warning("⚠️ Not enough data to run topic modeling (<5 events).")
        return "Not enough data."
    
    # 2. Run BERTopic
    logger.info(f"📊 Running BERTopic on {len(events)} events...")
    docs = [f"{ev.get('event','')} {ev.get('description','')}" for ev in events]
    embeddings_np = np.stack(embeddings)
    
    norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    embeddings_np = embeddings_np / norm

    topic_model = BERTopic(min_topic_size=3, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings_np)

    # 3. Update Postgres
    logger.info("💾 Saving topic assignments to Postgres...")
    with get_db_cursor() as cur:
        for ev, tid in zip(events, topics):
            cur.execute("UPDATE umd_events SET topic_id = %s WHERE id = %s", (int(tid), ev['id']))

    # 4. Sync Elasticsearch
    logger.info("🔄 Syncing topics to Elasticsearch...")
    success_count = 0
    for ev, tid in zip(events, topics):
        q = {
            "query": { "match_phrase": { "event": ev['event'] } },
            "script": { "source": "ctx._source.topic_id = params.tid", "params": {"tid": int(tid)} }
        }
        try:
            es_client.update_by_query(index="umd_events", body=q, conflicts="proceed")
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to sync event {ev['id']} to Elastic: {e}")

    # 5. Generate Labels (Calculated IN MEMORY first to avoid DB Locks)
    logger.info("🏷️ Generating Topic Labels (AI)...")
    topic_info = topic_model.get_topic_info()
    labels_to_save = [] # Store them here first

    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1: continue 
        
        keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]
        
        try:
            # AI Call happens HERE (No DB connection held)
            prompt = f"Keywords: {', '.join(keywords)}. Provide a concise category name (max 3 words). Return ONLY the name."
            resp = llm_client.chat.completions.create(
                model=LABEL_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=15
            )
            label = resp.choices[0].message.content.strip().replace('"','')
        except Exception as e:
            logger.error(f"Label generation failed for topic {tid}: {e}")
            label = f"Topic {tid}"
        
        # Add to list
        labels_to_save.append((int(tid), label, ", ".join(keywords)))

    # 6. Save Labels to DB (Fast Batch Insert)
    logger.info("💾 Saving labels to Database...")
    with get_db_cursor() as cur:
        cur.execute("TRUNCATE TABLE topic_labels")
        for tid, label, kw in labels_to_save:
            cur.execute("INSERT INTO topic_labels (topic_id, label, keywords) VALUES (%s, %s, %s)", 
                        (tid, label, kw))
    
    logger.info(f"✅ Pipeline Complete. Synced {success_count} events.")
    return f"Pipeline Complete. Synced {success_count} events to Elastic."

# ============================
#  CHAINLIT EVENT HANDLERS
# ============================

@cl.on_chat_start
async def start():
    init_db()
    
    start_msg = cl.Message(content="🐢 **TestudoBot is booting up...**")
    await start_msg.send()

    # Check for startup work
    with get_db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id IS NULL OR topic_id = -1;")
        uncategorized_count = cur.fetchone()[0]

    if uncategorized_count > 100:
        start_msg.content = f"🐢 **TestudoBot is booting up...**\n\n⚠️ Found {uncategorized_count} new/uncategorized events.\nRunning AI categorization pipeline (this may take 30s)..."
        await start_msg.update()
        
        # Run pipeline
        await cl.make_async(run_pipeline_if_needed)()
        
        start_msg.content = "✅ **Optimization Complete!**\nLoading interface..."
        await start_msg.update()

    # Create Settings
    topic_map = fetch_topic_map()
    topic_labels = ["All Topics"] + list(topic_map.keys())
    
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="topic_filter",
                label="Filter by Topic",
                values=topic_labels,
                initial_value="All Topics"
            ),
            cl.input_widget.Slider(
                id="top_k",
                label="Max Results",
                initial=20,
                min=1,
                max=30,
                step=1
            ),
        ]
    ).send()
    
    cl.user_session.set("topic_map", topic_map)
    cl.user_session.set("settings", {"topic_filter": "All Topics", "top_k": 5})
    cl.user_session.set("history", [])

    # Quick Actions
    actions = [
        cl.Action(name="quick_search", value="Is there free food today?", label="🍕 Free Food"),
        cl.Action(name="quick_search", value="Career fairs this month", label="💼 Career Fairs"),
        cl.Action(name="quick_search", value="Music performances next week", label="🎵 Music"),
        cl.Action(name="quick_search", value="Sports games this weekend", label="🐢 Sports"),
    ]

    start_msg.content = f"✅ **Ready!** I know about {len(topic_map)} categories of events.\n\nClick a button or type a query to start!"
    start_msg.actions = actions
    await start_msg.update()

@cl.action_callback("quick_search")
async def on_action(action: cl.Action):
    await cl.Message(content=action.value, author="User").send()
    await main(cl.Message(content=action.value))

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"⚙️ **Filter Updated:** {settings['topic_filter']}").send()

# ============================
#  RAGAS EVALUATION LOGIC
# ============================

def build_eval_samples() -> List[Dict[str, str]]:
    return [
        {"question": "Are there any career fairs happening this month?", "ground_truth": "Lists upcoming career fairs at UMD."},
        {"question": "What music performances are scheduled?", "ground_truth": "Summarizes music or concert events."},
        {"question": "Is there free food at any event?", "ground_truth": "Identifies events that explicitly mention free food."},
    ]

async def run_ragas_evaluation():
    eval_samples = build_eval_samples()
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    ragas_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        temperature=0.0
    )
    ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    for sample in eval_samples:
        q = sample["question"]
        gt = sample["ground_truth"]
        results = search_events(q, top_k=5)
        context_text = "\n".join([f"{ev.get('event','')} {ev.get('description','')}" for ev, score in results])
        ctx_list = [context_text] if context_text else [""]

        # Simple generation for evaluation
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": f"Context: {context_text}\nQuestion: {q}"}],
            temperature=0.1
        )
        ans = resp.choices[0].message.content

        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts_list.append(ctx_list)

    data = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    
    return evaluate(
        dataset=Dataset.from_dict(data),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

@cl.on_message
async def main(message: cl.Message):
    if message.content.strip() == "/test":
        await cl.Message(content="📊 **Starting RAGAS Evaluation...** (Check terminal)").send()
        results = await run_ragas_evaluation()
        df = results.to_pandas()
        csv_file = cl.File(name="ragas_results.csv", content=df.to_csv().encode("utf-8"))
        
        scores_dict = df.select_dtypes(include=[np.number]).mean().to_dict()
        summary = "\n".join([f"- **{k}**: {v:.4f}" for k, v in scores_dict.items()])
        
        await cl.Message(content=f"✅ **Evaluation Complete!**\n\n{summary}", elements=[csv_file]).send()
        return

    history = cl.user_session.get("history", [])
    settings = cl.user_session.get("settings")
    topic_map = cl.user_session.get("topic_map")
    
    selected_label = settings.get("topic_filter", "All Topics")
    top_k = int(settings.get("top_k", 5))
    max_results = top_k
    topic_list = ", ".join(topic_map.keys()) if topic_map else "General Events"

    filter_id = int(topic_map[selected_label]) if (selected_label != "All Topics" and selected_label in topic_map) else None

    # Search
    async with cl.Step(name="Searching UMD Events...", type="tool") as step:
        step.input = message.content
        results = search_events(message.content, top_k=top_k, filter_topic_id=filter_id)
        if results:
            step.output = "\n".join([f"- {ev.get('event','Unknown')} ({score:.2f})" for ev, score in results])
        else:
            step.output = "No matches found."

    # Build Context safely
    context_text = "\n\n".join([
        f"Event: {ev.get('event','N/A')}\nDate: {ev.get('date','N/A')}\nDesc: {(ev.get('description') or '')[:300]}"
        for ev, score in results
    ]) or "No events found."

    current_date = datetime.now()
    date_str = current_date.strftime("%B %Y")
    sys_prompt = f"""
    You are TestudoBot, a knowledgeable, friendly, and enthusiastic AI assistant for University of Maryland (UMD) events. Your goal: Help users discover lectures, career fairs, performances, workshops, and more – using ONLY the provided Context. Do not invent details, dates, or URLs. If Context lacks info, say so politely and suggest alternatives from data.

    Key Parameters:
    - Current Date: {date_str} (resolve relatives; for 'December', use Dec {current_date.year} unless specified).
    - Known Categories: {topic_list}
    - Max Results: {max_results} (list ALL relevant from Context, even if >{max_results}; prioritize & sort by date).

    Instructions:
    1. Think step-by-step: (a) Parse query for date (e.g., 'December' = Dec 1-31), topic ('basketball'), etc. (b) From Context, filter STRICTLY to matching date range (ignore non-Dec events). Sort by soonest date first. Include women's/men's variants. (c) Summarize top {max_results} (or all if fewer), even with 'N/A' desc—use event name for details.
    2. If few/no matches: 'I found X basketball events in December. For more, try specifying date/type.' Don't say 'couldn't find' if Context has any.
    3. If Context has non-matching dates (e.g., Nov/Jan), ignore them entirely.
    4. Maintain conversation...
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
    - Query: 'basketball in december' → 'Here are December basketball events: \n• **Men's vs. Wagner** \nDate/Time: Dec 2, 2025, 8pm \n... \n• **Men's vs. Old Dominion** \nDate/Time: Dec 28, 2025, 6pm \n... \nFound 3 total.'
    - Query: "Career events next week?" → "Upcoming Career Services events next week: \n• **Job Fair** \nDate/Time: Oct 10, 2-5pm \nLocation: Stamp Student Union \nDesc: Networking for students and grads. \n[Source: Event 123 (https://umd.edu/fair)]"
    - Query: "Virtual events tomorrow?" (None) → "No virtual events tomorrow, but here's a close in-person alternative on Oct 9. Interested in more options?"
    - Query: "Fun events this month?" (Broad) → "Diverse fun events this month: \n• **Concert Series** \nDate/Time: Oct 15, 7pm \nLocation: Clarice Smith Center \nDesc: Live music performances. \n[Source: Event 456] \nPrefer a specific genre?"

    User Query (with history): {message.content}
    Context: {context_text}"""

    msg = cl.Message(content="")
    await msg.send()

    stream = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {message.content}"}
        ],
        temperature=0.1,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            await msg.stream_token(token)

    # Attach Sources (Rich Cards)
    if results:
        elements = []
        for i, (ev, score) in enumerate(results[:5]):
            lines = [
                f"**📅 When:** {ev.get('date', 'N/A')}" + (f" at {ev['time']}" if ev.get('time') else ""),
                f"**📍 Where:** {ev.get('location', 'TBD')}"
            ]
            if ev.get('url'):
                lines.append(f"**🔗 Link:** [Official Event Page]({ev['url']})")
            
            desc = (ev.get('description') or '')[:350]
            if len(ev.get('description') or '') > 350: desc += "..."
            lines.append(f"\n**📝 Details:**\n{desc}")
            
            elements.append(cl.Text(name=f"{i+1}. {ev.get('event','Unknown')}", content="\n".join(lines), display="inline"))
        msg.elements = elements
    
    await msg.update()
    
    # Save to history with FULL response
    history.append({"user": message.content, "bot": full_response}) 
    cl.user_session.set("history", history)