import os
import json
import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "Armani%40567")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "startup_db")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def create_embedding_column():
    """Add embedding column to organizations table"""
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE organizations 
                ADD COLUMN embedding_vector TEXT
            """))
            conn.commit()
            print("✅ Embedding column created")
        except Exception as e:
            print(f"⚠️ Column might already exist: {e}")

def generate_embeddings_batch():
    """Generate embeddings for all organizations"""
    
    # Load organizations
    with engine.connect() as conn:
        df = pd.read_sql("SELECT id, name, description, domain, skills, tags FROM organizations", conn)
    
    print(f"📊 Processing {len(df)} organizations...")
    
    # Create combined text for each organization
    df["combined_text"] = (
        df["name"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["domain"].fillna("") + " " +
        df["skills"].fillna("") + " " +
        df["tags"].fillna("")
    )
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            try:
                # Generate embedding using OpenAI's best model
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=row["combined_text"]
                )
                
                embedding = response.data[0].embedding
                embedding_json = json.dumps(embedding)
                
                # Store in database
                with engine.connect() as conn:
                    conn.execute(
                        text("""
                            UPDATE organizations 
                            SET embedding_vector = :embedding 
                            WHERE id = :id
                        """),
                        {"embedding": embedding_json, "id": row["id"]}
                    )
                    conn.commit()
                
                # Rate limiting - OpenAI allows 3000 RPM on tier 1
                time.sleep(0.02)  # 50 requests per second
                
            except Exception as e:
                print(f"❌ Error processing org {row['id']}: {e}")
                continue
    
    print("✅ All embeddings generated and stored!")

def verify_embeddings():
    """Check how many organizations have embeddings"""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN embedding_vector IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings
            FROM organizations
        """))
        row = result.fetchone()
        print(f"\n📈 Statistics:")
        print(f"   Total organizations: {row[0]}")
        print(f"   With embeddings: {row[1]}")
        print(f"   Coverage: {(row[1]/row[0]*100):.1f}%")

if __name__ == "__main__":
    print("🚀 Starting embedding generation process...\n")
    
    # Step 1: Create column
    create_embedding_column()
    
    # Step 2: Generate embeddings
    generate_embeddings_batch()
    
    # Step 3: Verify
    verify_embeddings()
    
    print("\n✅ Process complete!")
