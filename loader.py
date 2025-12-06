import os
import json
import time
import sys
import psycopg2
from psycopg2 import OperationalError
from sentence_transformers import SentenceTransformer
import numpy as np

# Flush prints for Docker
def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def load_events_from_json(path: str):
    print_flush(f"🔎 Looking for JSON at: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of event objects (list[dict])")

    print_flush(f"✅ Loaded {len(data)} events from JSON.")
    return data

def get_pg_connection_with_retry(max_attempts: int = 20, delay_sec: float = 2.0):
    dbname = os.getenv("POSTGRES_DB", "umd_events")
    user = os.getenv("POSTGRES_USER", "umd_user")
    password = os.getenv("POSTGRES_PASSWORD", "umd_password")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")

    for attempt in range(1, max_attempts + 1):
        try:
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
            )
            print_flush(f"✅ Connected to Postgres on attempt {attempt}")
            return conn
        except OperationalError as e:
            print_flush(
                f"⏳ Postgres not ready yet (attempt {attempt}/{max_attempts}): "
                f"{e.__class__.__name__}: {e}"
            )
            if attempt == max_attempts:
                print_flush("❌ Giving up connecting to Postgres.")
                raise
            time.sleep(delay_sec)

def load_to_postgres(events):
    if not events:
        print_flush("❌ No events to load. Exiting.")
        return

    print_flush("🔄 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    conn = get_pg_connection_with_retry()
    conn.autocommit = True
    cur = conn.cursor()

    # Step 1: Check and create pgvector extension
    print_flush("🔄 Checking pgvector extension...")
    try:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print_flush("✅ pgvector extension already installed.")
        else:
            cur.execute("CREATE EXTENSION vector;")
            print_flush("✅ pgvector extension created.")
    except Exception as e:
        print_flush(f"❌ pgvector extension failed: {e}")
        # If image is wrong, error will propagate
        raise

    # Step 2: Check if table exists and create if not
    print_flush("🔄 Ensuring table schema...")
    try:
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'umd_events');")
        table_exists = cur.fetchone()[0]
        if not table_exists:
            cur.execute("""
                CREATE TABLE umd_events (
                    id SERIAL PRIMARY KEY,
                    event TEXT,
                    date TEXT,
                    time TEXT,
                    url TEXT,
                    location TEXT,
                    description TEXT,
                    topic_id INTEGER DEFAULT -1,
                    embedding VECTOR(384)
                );
            """)
            print_flush("✅ Table created fresh.")
        else:
            print_flush("📊 Table already exists; evolving schema...")
    except Exception as e:
        print_flush(f"❌ Table check/creation failed: {e}")
        raise

    # Step 3: Evolve schema (add missing columns)
    force_clean = os.getenv("FORCE_CLEAN_SCHEMA", "true") == "true"
    try:
        # Add topic_id
        cur.execute("""
            ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS topic_id INTEGER DEFAULT -1;
        """)
        print_flush("✅ topic_id column ensured.")

        # Add embedding (depends on vector type)
        cur.execute("""
            ALTER TABLE umd_events ADD COLUMN IF NOT EXISTS embedding VECTOR(384);
        """)
        print_flush("✅ embedding column ensured.")
    except Exception as e:
        print_flush(f"❌ ALTER failed: {e} (likely vector type missing)")
        if force_clean:
            print_flush("🔄 Fallback: Dropping table and recreating for clean schema...")
            cur.execute("DROP TABLE IF EXISTS umd_events CASCADE;")
            cur.execute("""
                CREATE TABLE umd_events (
                    id SERIAL PRIMARY KEY,
                    event TEXT,
                    date TEXT,
                    time TEXT,
                    url TEXT,
                    location TEXT,
                    description TEXT,
                    topic_id INTEGER DEFAULT -1,
                    embedding VECTOR(384)
                );
            """)
            print_flush("✅ Table recreated cleanly.")
        else:
            raise

    # Step 4: Verify columns (critical check)
    print_flush("🔄 Verifying schema...")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'umd_events' AND column_name IN ('topic_id', 'embedding');
    """)
    cols = {row[0]: row[1] for row in cur.fetchall()}
    
    # --- FIX 1: Allow 'USER-DEFINED' as a valid type for embedding ---
    # Postgres usually reports 'vector' as 'USER-DEFINED' in information_schema
    if 'embedding' not in cols or cols['embedding'] not in ('vector', 'USER-DEFINED'):
        raise ValueError(f"❌ Critical: embedding column missing or wrong type! Got: {cols}")
    
    print_flush(f"✅ Schema verified: topic_id ({cols.get('topic_id')}), embedding ({cols.get('embedding')}).")

    # Step 5: TRUNCATE
    cur.execute("TRUNCATE TABLE umd_events;")
    print_flush("🗑️ Old data cleared.")

    # Step 6: Compute embeddings
    print_flush("🔄 Computing embeddings...")
    texts = [f"{ev.get('event', '')} {ev.get('date', '')} {ev.get('time', '')} {ev.get('url', '')} {ev.get('location', '')} {ev.get('description', '')}".strip() for ev in events]
    embeddings = model.encode(texts, normalize_embeddings=True).astype('float32')
    print_flush(f"✅ Computed {len(embeddings)} embeddings (shape: {embeddings.shape}).")

    # Step 7: Insert (batch-like loop with error per-row)
    insert_sql = """
        INSERT INTO umd_events (event, date, time, url, location, description, topic_id, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    
    inserted_count = 0
    for i, ev in enumerate(events):
        # --- FIX 2: Convert to Python LIST (not bytes) for pgvector ---
        embedding_list = embeddings[i].tolist()
        
        try:
            cur.execute(insert_sql, (
                ev.get("event") or None,
                ev.get("date") or None,
                ev.get("time") or None,
                ev.get("url") or None,
                ev.get("location") or None,
                ev.get("description") or None,
                -1,  # Default
                embedding_list, # Passing list[float]
            ))
            cur.fetchone()  # Consume RETURNING
            inserted_count += 1
        except Exception as e:
            print_flush(f"❌ Insert failed for event {i+1}: {e}")
            # Continue to insert others, but raise at end if any failed
    print_flush(f"📝 Inserted {inserted_count}/{len(events)} rows.")

    if inserted_count != len(events):
        raise ValueError("Partial insert failure – check logs.")

    # Step 8: Create index
    try:
        cur.execute("DROP INDEX IF EXISTS umd_events_embedding_idx;")
        cur.execute("""
            CREATE INDEX umd_events_embedding_idx 
            ON umd_events USING hnsw (embedding vector_cosine_ops);
        """)
        print_flush("🗂️ HNSW index created.")
    except Exception as e:
        print_flush(f"⚠️ Index creation warning: {e}")

    cur.close()
    conn.close()
    print_flush(f"🎉 Loaded {len(events)} events into Postgres with embeddings successfully!")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_filename = os.getenv("EVENTS_JSON_NAME", "umd_calendar_2025-10-01_to_2025-10-31.json")
    json_path = os.path.join(base_dir, json_filename)
    print_flush(f"📂 Loading events from: {json_path}")

    events = load_events_from_json(json_path)
    load_to_postgres(events)

if __name__ == "__main__":
    main()