# app.py
import os
import json
import time
import logging
import chainlit as cl
from datetime import datetime
from typing import List, Dict
import numpy as np # Helper import for pipeline
import psycopg2
from sentence_transformers import SentenceTransformer

from psycopg2 import pool
from elasticsearch import Elasticsearch
# from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
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
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "google/gemini-3-pro-preview"  # <--- The model you want
SITE_URL = os.getenv("OR_SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("OR_SITE_NAME", "TestudoBot")
# --- GLOBAL CLIENTS ---
# In Chainlit, it's safe to keep these global as they are thread-safe/stateless
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

from contextlib import contextmanager

# 1. Create a Global Connection Pool (Runs once on startup)
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20, # Min 1, Max 20 connections
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432"
    )
    if db_pool:
        logger.info("✅ Database connection pool created successfully")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")

# 2. Helper to get a cursor from the pool
@contextmanager
def get_db_cursor(cursor_factory=None):
    """
    Yields a cursor from a pooled connection, then automatically puts the connection back.
    Usage:
        with get_db_cursor() as cur:
            cur.execute(...)
    """
    conn = db_pool.getconn()
    try:
        # Support for DictCursor or standard cursor
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
    # conn = get_db_connection()
    try:
        with get_db_cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS topic_id INTEGER;")
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
        
    except Exception as e:
        logger.error(f"DB Init Error: {e}")
        conn.rollback()
 

def fetch_topic_map() -> Dict[str, str]:
    """Returns { 'Career': '1', 'Music': '2' } for the dropdown."""
    # conn = get_db_connection()
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT topic_id, label FROM topic_labels WHERE topic_id != -1 ORDER BY label;")
            # Return Label -> ID mapping for the UI
            return {row[1]: str(row[0]) for row in cur.fetchall()}
    except Exception:
        return {}
 

def search_events(query_text, top_k=5, filter_topic_id=None):
    # 1. Vector Search
    query_vector = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
    
    filters = []
    if filter_topic_id is not None:
        filters.append({"term": {"topic_id": filter_topic_id}})

    search_body = {
        "size": top_k,
        "knn": {
            "field": "embedding", 
            "query_vector": query_vector,
            "k": top_k, 
            "num_candidates": 50, 
            "boost": 0.5, 
            "filter": filters
        },
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query_text, 
                        "fields": ["event^2", "description", "location"], 
                        "boost": 0.5
                    }
                },
                "filter": filters
            }
        }
    }

    try:
        response = es_client.search(index="umd_events", body=search_body)
        return [(hit['_source'], hit['_score']) for hit in response['hits']['hits']]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

# --- PIPELINE & ADMIN TASKS ---

# def run_pipeline_if_needed():
#     """Checks for uncategorized events and runs the pipeline."""
#     conn = get_db_connection()
#     uncategorized_count = 0
#     try:
#         with conn.cursor() as cur:
#             cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id = -1;")
#             result = cur.fetchone()
#             if result:
#                 uncategorized_count = result[0]
#     finally:
#         conn.close()

#     topic_map = fetch_topic_map()
    
#     # If no topics OR we have new data, run the expensive pipeline
#     if not topic_map or uncategorized_count > 0:
#         logger.info(f"⚠️ Found {uncategorized_count} uncategorized events. Running Pipeline...")
#         return run_topic_modeling_pipeline()
#     return False

def run_pipeline_if_needed():
    """Checks if we need to run the pipeline."""
    # We don't even need to check uncategorized_count anymore
    # because your data is fixed.
    
    topic_map = fetch_topic_map()
    
    # --- CHANGED LOGIC ---
    # OLD: if not topic_map or uncategorized_count > 0:
    # NEW: Only run if the topic map is EMPTY (meaning it's the very first run)
    if not topic_map:
        logger.info(f"⚠️ First-time setup detected. Running AI Pipeline...")
        return run_topic_modeling_pipeline()
        
    logger.info("✅ Topics already exist. Skipping pipeline.")
    return False

def run_topic_modeling_pipeline():
    """
    Runs BERTopic -> Updates Postgres -> Syncs Elasticsearch -> Generates Labels
    """
    from bertopic import BERTopic
    from psycopg2.extras import DictCursor
    

    # 1. Fetch Data
   # conn = get_db_connection()
    events = []
    embeddings = []
    try:
        with get_db_cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT id, event, description, embedding FROM umd_events")
            rows = cur.fetchall()
            for r in rows:
                if r['embedding'] is None: continue
                # Parse embedding depending on if it's string or list
                emb = np.array(json.loads(r['embedding']) if isinstance(r['embedding'], str) else r['embedding'])
                if np.count_nonzero(emb) == 0: continue
                events.append(dict(r))
                embeddings.append(emb)
    finally:
        pass

    if len(events) < 5: return "Not enough data."
    
    # 2. Run BERTopic
    docs = [f"{ev.get('event','')} {ev.get('description','')}" for ev in events]
    embeddings_np = np.stack(embeddings)
    
    # Normalize embeddings for better clustering
    norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    embeddings_np = embeddings_np / norm

    topic_model = BERTopic(min_topic_size=3, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings_np)

    # 3. Update Postgres (The Source of Truth)
    try:
        with get_db_cursor() as cur:
            for ev, tid in zip(events, topics):
                cur.execute("UPDATE umd_events SET topic_id = %s WHERE id = %s", (int(tid), ev['id']))
        conn.commit()
    finally:
        pass

    # 4. Sync Elasticsearch (CRITICAL STEP)
    # We update Elastic so the "Filter by Topic" feature works in search
    success_count = 0
    for ev, tid in zip(events, topics):
        # We find the document by exact event name match and update its topic_id
        q = {
            "query": { "match_phrase": { "event": ev['event'] } },
            "script": { "source": "ctx._source.topic_id = params.tid", "params": {"tid": int(tid)} }
        }
        try:
            es_client.update_by_query(index="umd_events", body=q, conflicts="proceed")
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to sync event {ev['id']} to Elastic: {e}")

    # 5. Generate & Save Labels (Using Groq)
    topic_info = topic_model.get_topic_info()
    try:
        with get_db_cursor() as cur:
            cur.execute("TRUNCATE TABLE topic_labels")
            for _, row in topic_info.iterrows():
                tid = row['Topic']
                if tid == -1: continue # Skip noise
                
                # Get top 5 keywords for this topic
                keywords = [x[0] for x in topic_model.get_topic(tid)[:5]]
                
                # Generate Label via Groq
                try:
                    time.sleep(1.0) # Rate limit safety
                    prompt = f"Keywords: {', '.join(keywords)}. Provide a concise category name (max 3 words). Return ONLY the name."
                    resp = llm_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=15
                    )
                    label = resp.choices[0].message.content.strip().replace('"','')
                except Exception as e:
                    logger.error(f"Label generation failed for topic {tid}: {e}")
                    label = f"Topic {tid}"
                
                cur.execute("INSERT INTO topic_labels (topic_id, label, keywords) VALUES (%s, %s, %s)", 
                            (int(tid), label, ", ".join(keywords)))
        conn.commit()
    finally:
        pass
        
    return f"Pipeline Complete. Synced {success_count} events to Elastic."



# ============================
#  CHAINLIT EVENT HANDLERS
# ============================

@cl.on_chat_start
async def start():
    # 1. Initial Setup
    init_db()
    
    # 2. Check Startup
    start_msg = cl.Message(content="🐢 **TestudoBot is booting up...**")
    await start_msg.send()

    # CHECK: Is this the first run?
    with get_db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM umd_events WHERE topic_id = -1;")
        uncategorized_count = cur.fetchone()[0]

    if uncategorized_count > 0:
        # --- FIX 1: Set content property, then update ---
        start_msg.content = f"🐢 **TestudoBot is booting up...**\n\n⚠️ Found {uncategorized_count} new events.\nRunning AI categorization pipeline (this may take 30s)..."
        await start_msg.update()
        
        # Run the heavy pipeline
        await cl.make_async(run_pipeline_if_needed)()
        
        # --- FIX 2: Set content property, then update ---
        start_msg.content = "✅ **Optimization Complete!**\nLoading interface..."
        await start_msg.update()

    # 3. Create Settings Menu
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
                initial=5,
                min=1,
                max=10,
                step=1
            ),
        ]
    ).send()
    
    cl.user_session.set("topic_map", topic_map)
    cl.user_session.set("settings", {"topic_filter": "All Topics", "top_k": 5})

    # 4. Initialize History (Critical for Memory)
    cl.user_session.set("history", [])

    # 5. Define Quick Actions
    actions = [
        cl.Action(name="quick_search", value="Is there free food today?", label="🍕 Free Food"),
        cl.Action(name="quick_search", value="Career fairs this month", label="💼 Career Fairs"),
        cl.Action(name="quick_search", value="Music performances next week", label="🎵 Music"),
        cl.Action(name="quick_search", value="Sports games this weekend", label="🐢 Sports"),
    ]

    # 6. Final Update with Actions
    # We update the existing start_msg instead of sending a new one to keep the chat clean
    start_msg.content = f"✅ **Ready!** I know about {len(topic_map)} categories of events.\n\nClick a button or type a query to start!"
    start_msg.actions = actions
    await start_msg.update()

    # --- NEW CALLBACK FUNCTION ---
    # (Paste this OUTSIDE and BELOW the start() function)
    @cl.action_callback("quick_search")
    async def on_action(action: cl.Action):
        # This simulates the user typing the label text
        await cl.Message(content=action.value, author="User").send()
        # Pass the fake message to your main function to trigger the search
        await main(cl.Message(content=action.value))


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    # Give feedback that settings changed
    await cl.Message(content=f"⚙️ **Filter Updated:** {settings['topic_filter']}").send()

# ============================
#  RAGAS EVALUATION LOGIC
# ============================

def build_eval_samples() -> List[Dict[str, str]]:
    """Defines the questions and ideal answers for testing."""
    return [
        {
            "question": "Are there any career fairs happening this month?",
            "ground_truth": "Lists upcoming career fairs at UMD with dates and locations.",
        },
        {
            "question": "What music performances are scheduled?",
            "ground_truth": "Summarizes music or concert events at The Clarice or other venues.",
        },
        {
            "question": "Is there free food at any event?",
            "ground_truth": "Identifies events that explicitly mention free food or catering.",
        }
    ]

async def run_ragas_evaluation():
    """
    Runs RAGAS metrics using your OpenRouter connection.
    """
    eval_samples = build_eval_samples()
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    # 1. RAGAS needs an LLM to act as the "Judge". 
    # We configure LangChain to use your OpenRouter key.
    ragas_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL, # Uses Gemini 3 as the judge
        temperature=0.0
    )
    
    # 2. Embeddings for metrics calculation
    ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. Generate Answers for each test question
    for sample in eval_samples:
        q = sample["question"]
        gt = sample["ground_truth"]
        
        # A. Retrieve (Uses your NEW search_events function)
        results = search_events(q, top_k=5)
        
        # B. Format Context
        context_text = "\n".join([f"{ev['event']} {ev['description']}" for ev, score in results])
        ctx_list = [context_text] if context_text else [""]

        # C. Generate Answer (We assume a simple non-streaming call for testing)
        sys_prompt = f"Answer based on context: {context_text}"
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q}
            ],
            temperature=0.1
        )
        ans = resp.choices[0].message.content

        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts_list.append(ctx_list)

    # 4. Build Dataset
    data = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 5. Run Evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    return results

@cl.on_message
async def main(message: cl.Message):
    if message.content.strip() == "/test":
        await cl.Message(content="📊 **Starting RAGAS Evaluation...** (Check terminal for progress)").send()
        
        # Run the evaluation (this might take a minute)
        results = await run_ragas_evaluation()
        print(results)
# Display Results
        df = results.to_pandas()
        csv_file = cl.File(name="ragas_results.csv", content=df.to_csv().encode("utf-8"))
        
        # --- FIX: Convert to dict first ---
        scores_dict = df.select_dtypes(include=[np.number]).mean().to_dict()
        summary = "\n".join([f"- **{k}**: {v:.4f}" for k, v in scores_dict.items()])
        
        await cl.Message(
            content=f"✅ **Evaluation Complete!**\n\n{summary}",
            elements=[csv_file]
        ).send()
        return
    history = cl.user_session.get("history", [])
    # 1. Retrieve Settings
    settings = cl.user_session.get("settings")
    topic_map = cl.user_session.get("topic_map")
    
    selected_label = settings["topic_filter"]
    top_k = int(settings["top_k"])
    
    # --- FIX START: Define variables for the Prompt ---
    # Create a string list of topics for the LLM (e.g., "Career, Music, Sports")
    topic_list = ", ".join(topic_map.keys()) if topic_map else "General Events"
    max_results = top_k
    # --- FIX END ---

    # Convert Label ("Career") -> ID (1)
    filter_id = None
    if selected_label != "All Topics" and selected_label in topic_map:
        filter_id = int(topic_map[selected_label])

    # 2. Show "Thinking..." Step
    async with cl.Step(name="Searching UMD Events...", type="tool") as step:
        step.input = message.content
        results = search_events(message.content, top_k=top_k, filter_topic_id=filter_id)
        
        if results:
            step.output = "\n".join([f"- {ev['event']} ({score:.2f})" for ev, score in results])
        else:
            step.output = "No matches found."

    # 3. Generate Answer (Streaming)
    context_text = "\n\n".join([
        f"Event: {ev['event']}\nDate: {ev['date']}\nDesc: {ev['description'][:300]}"
        for ev, score in results
    ]) or "No events found."
    history_text = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[:]])
    current_date = datetime.now().strftime("%B %Y")
    sys_prompt = f"""
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

    User Query (with history): {message.content}
    Context: {context_text}"""

    msg = cl.Message(content="")
    await msg.send()

    stream = llm_client.chat.completions.create(
    model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuery: {message.content}"}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            await msg.stream_token(chunk.choices[0].delta.content)

# 4. Attach Sources to the final message
# 4. Attach Sources to the final message (Rich Cards)# 4. Attach Sources to the final message (Rich Cards)
# 4. Attach Sources to the final message (Smart Rich Cards)
    if results:
        elements = []
        for i, (ev, score) in enumerate(results):
            
            # --- Build the Card Content Dynamically ---
            lines = []
            
            # 1. Date & Time (Combine if time exists)
            date_str = ev.get('date', 'N/A')
            if ev.get('time'):
                date_str += f" at {ev['time']}"
            lines.append(f"**📅 When:** {date_str}")
            
            # 2. Location
            lines.append(f"**📍 Where:** {ev.get('location', 'TBD')}")
            
            # 3. URL (Only add if it exists)
            if ev.get('url'):
                lines.append(f"**🔗 Link:** [Official Event Page]({ev['url']})")
            
            # 4. Score (Good for debugging, maybe optional for users)
            lines.append(f"**📊 Match:** {score:.2f}")
            
            # 5. Description (Expanded to 350 chars to catch contact info)
            desc = ev.get('description', '')[:350]
            # Add "..." if we cut it off
            if len(ev.get('description', '')) > 350:
                desc += "..."
            
            lines.append(f"\n**📝 Details:**\n{desc}")

            # Join lines with newlines
            source_content = "\n".join(lines)

            # --- Create the Element ---
            elements.append(
                cl.Text(
                    name=f"{i+1}. {ev['event']}", 
                    content=source_content,
                    display="inline"
                )
            )
        msg.elements = elements
    await msg.update()
    history.append({"user": message.content, "bot": msg.content}) 
    cl.user_session.set("history", history)