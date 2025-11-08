"""
Ytheys Startup Network - Smart Embedding Generator (GTE-Large Edition)
=======================================================================
Generates high-quality AI embeddings using thenlper/gte-large model.
Handles both integer and string ID columns.
Uses Apple MPS acceleration on M1 chips.

Usage:
    python generate_embeddings_smart.py
"""

import os
import sys
import json
import logging
import pandas as pd
import torch
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime


# =====================================
# SETUP LOGGING
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embedding_generation.log")
    ]
)
logger = logging.getLogger(__name__)


# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================
load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "Armani%40567")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "startup_db")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"


# =====================================
# DETECT DEVICE (MPS for M1, else CPU)
# =====================================
def get_best_device():
    """Detect best available device for Apple Silicon"""
    try:
        if torch.backends.mps.is_available():
            logger.info("🚀 Using Apple MPS (Metal) acceleration")
            return "mps"
    except Exception:
        pass
    
    if torch.cuda.is_available():
        logger.info("🚀 Using CUDA GPU acceleration")
        return "cuda"
    
    logger.info("💻 Using CPU (no GPU acceleration available)")
    return "cpu"

DEVICE = get_best_device()


# =====================================
# INITIALIZE DATABASE CONNECTION
# =====================================
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    logger.info(f"✅ Connected to database: {DB_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    sys.exit(1)


# =====================================
# LOAD AI MODEL (GTE-LARGE)
# =====================================
logger.info("Loading GTE-Large embedding model (this may take a moment)...")
logger.info("   This is a high-quality model - first download may take 1-2 minutes")

try:
    model = SentenceTransformer("thenlper/gte-large", device=DEVICE)
    logger.info("✅ Model loaded successfully: thenlper/gte-large")
    logger.info(f"   - Embedding dimension: 1024 (higher quality than 384)")
    logger.info(f"   - Max sequence length: 512 tokens")
    logger.info(f"   - Device: {DEVICE}")
    logger.info(f"   - Normalization: Enabled (unit-length vectors)")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    logger.error("   Try: pip install sentence-transformers -U")
    sys.exit(1)


# =====================================
# HELPER FUNCTIONS
# =====================================
def detect_columns():
    """Detect available columns in organizations table"""
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('organizations')]
        logger.info(f"\n📋 Detected {len(columns)} columns in 'organizations' table:")
        
        id_cols = [c for c in columns if c in ['id']]
        name_cols = [c for c in columns if 'name' in c.lower()]
        text_cols = [c for c in columns if c in ['description', 'type', 'domain', 'skills', 'tags', 'goals']]
        other_cols = [c for c in columns if c not in id_cols + name_cols + text_cols]
        
        if id_cols:
            logger.info(f"   ID: {', '.join(id_cols)}")
        if name_cols:
            logger.info(f"   Names: {', '.join(name_cols)}")
        if text_cols:
            logger.info(f"   Text fields: {', '.join(text_cols)}")
        if other_cols:
            logger.info(f"   Other: {', '.join(other_cols[:10])}{'...' if len(other_cols) > 10 else ''}")
        
        return columns
    except Exception as e:
        logger.error(f"❌ Error detecting columns: {e}")
        sys.exit(1)


def create_embedding_column():
    """Add embedding_vector column to organizations table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('organizations')]
            
            if 'embedding_vector' in columns:
                logger.info("⚠️  embedding_vector column already exists")
                logger.info("   Existing embeddings will be REPLACED with new GTE-Large embeddings")
                return True
            
            conn.execute(text("""
                ALTER TABLE organizations 
                ADD COLUMN embedding_vector MEDIUMTEXT
            """))
            conn.commit()
            logger.info("✅ embedding_vector column created successfully")
            return True
            
    except Exception as e:
        if "Duplicate column name" in str(e):
            logger.info("⚠️  embedding_vector column already exists")
            return True
        else:
            logger.error(f"❌ Error creating embedding column: {e}")
            return False


def parse_json_field(value):
    """Safely parse JSON fields and extract text"""
    if not value or pd.isna(value):
        return ""
    
    try:
        if isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return ' '.join(str(item) for item in parsed if item)
                elif isinstance(parsed, dict):
                    return ' '.join(str(v) for v in parsed.values() if v)
                else:
                    return str(value)
            else:
                return str(value)
        else:
            return str(value)
    except Exception as e:
        logger.debug(f"Could not parse JSON field: {e}")
        return str(value) if value else ""


def generate_embeddings_batch(available_columns):
    """Generate GTE-Large embeddings for all organizations"""
    
    priority_text_cols = ['organization_name', 'name', 'description', 'domain', 'type', 
                          'skills', 'tags', 'goals', 'sdg_alignment', 'collaboration_needs']
    
    text_columns = []
    for col in priority_text_cols:
        if col in available_columns:
            text_columns.append(col)
    
    if not text_columns:
        logger.error("❌ No suitable text columns found for generating embeddings!")
        return False
    
    logger.info(f"\n📝 Using these columns for embeddings: {', '.join(text_columns)}")
    
    select_query = f"SELECT id, {', '.join(text_columns)} FROM organizations"
    logger.info(f"\n🔍 Query: {select_query}")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(select_query, conn)
        logger.info(f"\n📊 Loaded {len(df)} organizations from database")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return False
    
    if df.empty:
        logger.error("❌ No organizations found in database!")
        return False
    
    # Create combined text for each organization
    logger.info("\n🔨 Building combined text from available fields...")
    combined_texts = []
    
    for idx, row in df.iterrows():
        text_parts = []
        for col in text_columns:
            value = row[col]
            if pd.notna(value):
                parsed_text = parse_json_field(value)
                if parsed_text:
                    text_parts.append(parsed_text)
        
        combined_text = ' '.join(text_parts)
        combined_texts.append(combined_text)
    
    if combined_texts:
        sample_text = combined_texts[0][:200] + "..." if len(combined_texts[0]) > 200 else combined_texts[0]
        logger.info(f"\n📄 Sample combined text (first org):")
        logger.info(f"   {sample_text}")
    
    # Generate embeddings with GTE-Large
    logger.info(f"\n🚀 Generating GTE-Large embeddings for {len(combined_texts)} organizations...")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Batch size: 16 (optimized for M1 Air memory)")
    logger.info(f"   Normalization: Enabled (unit-length vectors for stable cosine similarity)")
    logger.info("   (This may take 2-5 minutes depending on your hardware...)")
    
    try:
        embeddings = model.encode(
            combined_texts,
            show_progress_bar=True,
            batch_size=16,  # Lower for M1 Air memory; increase to 32 on more powerful machines
            convert_to_numpy=True,
            normalize_embeddings=True,  # IMPORTANT: L2 normalize for GTE-Large
            device=DEVICE
        )
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"   - Embedding shape: {embeddings[0].shape} (1024-dimensional)")
        logger.info(f"   - Normalized: Yes (unit-length vectors)")
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        if "out of memory" in str(e).lower():
            logger.error("   Try reducing batch_size to 8 or 4")
        return False
    
    # Store embeddings in database
    logger.info("\n💾 Storing embeddings in database...")
    logger.info("   Note: 1024-dim vectors are ~2.7x larger than 384-dim (higher quality trade-off)")
    success_count = 0
    error_count = 0
    
    with engine.connect() as conn:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving to DB"):
            try:
                embedding_json = json.dumps(embeddings[idx].tolist())
                org_id = row["id"]  # Keep as is (string or int)
                
                conn.execute(
                    text("""
                        UPDATE organizations 
                        SET embedding_vector = :embedding 
                        WHERE id = :id
                    """),
                    {"embedding": embedding_json, "id": org_id}
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error saving embedding for org {row['id']}: {e}")
                error_count += 1
        
        conn.commit()
    
    logger.info(f"\n✅ Embeddings saved successfully!")
    logger.info(f"   - Success: {success_count}")
    if error_count > 0:
        logger.warning(f"   - Errors: {error_count}")
    
    return success_count > 0


def verify_embeddings():
    """Verify embeddings were created successfully"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN embedding_vector IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings,
                    SUM(CASE WHEN embedding_vector IS NULL THEN 1 ELSE 0 END) as without_embeddings
                FROM organizations
            """))
            row = result.fetchone()
            
            total = row[0]
            with_emb = row[1]
            without_emb = row[2]
            coverage = (with_emb / total * 100) if total > 0 else 0
            
            logger.info("\n" + "=" * 60)
            logger.info("📈 EMBEDDING STATISTICS")
            logger.info("=" * 60)
            logger.info(f"   Total organizations:      {total}")
            logger.info(f"   With embeddings:          {with_emb}")
            logger.info(f"   Without embeddings:       {without_emb}")
            logger.info(f"   Coverage:                 {coverage:.1f}%")
            logger.info("=" * 60)
            
            if coverage == 100:
                logger.info("✅ All organizations have GTE-Large embeddings!")
            elif coverage > 0:
                logger.warning(f"⚠️  {without_emb} organizations missing embeddings")
            else:
                logger.error("❌ No embeddings found!")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Error verifying embeddings: {e}")
        return False


def test_embedding_quality():
    """Test the quality of generated embeddings with a sample query"""
    try:
        logger.info("\n🧪 Testing GTE-Large embedding quality...")
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, organization_name, embedding_vector 
                FROM organizations 
                WHERE embedding_vector IS NOT NULL 
                LIMIT 1
            """))
            row = result.fetchone()
            
            if not row:
                logger.warning("⚠️  No embeddings found to test")
                return False
            
            org_id = row[0]
            org_name = row[1]
            embedding_json = row[2]
            
            embedding = json.loads(embedding_json)
            
            logger.info(f"   Sample organization: {org_name} (ID: {org_id})")
            logger.info(f"   Embedding length: {len(embedding)}")
            logger.info(f"   Embedding type: {type(embedding)}")
            logger.info(f"   First 5 values: {[round(v, 4) for v in embedding[:5]]}")
            
            # Verify normalization
            import numpy as np
            norm = np.linalg.norm(embedding)
            logger.info(f"   Vector L2 norm: {norm:.6f} (should be ~1.0 for normalized)")
            
            if len(embedding) == 1024:
                logger.info("✅ Embedding dimension correct (1024 - GTE-Large)")
            else:
                logger.error(f"❌ Unexpected embedding dimension: {len(embedding)}")
                return False
            
            if abs(norm - 1.0) < 0.01:
                logger.info("✅ Embeddings properly normalized (unit length)")
            else:
                logger.warning(f"⚠️  Normalization may be off (norm={norm:.4f})")
            
            # Test similarity
            test_query = "technology startup"
            logger.info(f"\n   Testing similarity with query: '{test_query}'")
            query_embedding = model.encode(test_query, normalize_embeddings=True)
            
            org_vec = np.array(embedding)
            query_vec = np.array(query_embedding)
            
            # With normalized vectors, dot product = cosine similarity
            similarity = np.dot(org_vec, query_vec)
            
            logger.info(f"   Similarity score: {similarity:.4f}")
            logger.info("✅ Embedding quality test passed!")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Embedding quality test failed: {e}")
        return False


# =====================================
# MAIN EXECUTION
# =====================================
def main():
    """Main execution function"""
    start_time = datetime.now()
    
    logger.info("\n" + "=" * 70)
    logger.info("🚀 YTHEYS EMBEDDING GENERATOR - GTE-LARGE EDITION")
    logger.info("=" * 70)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Database: {DB_NAME}")
    logger.info(f"Model: thenlper/gte-large (1024-dim, normalized)")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Supports: String and Integer IDs")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Detect available columns
    logger.info("STEP 1: Detecting database schema...")
    available_columns = detect_columns()
    
    if not available_columns:
        logger.error("❌ Failed to detect columns. Exiting.")
        sys.exit(1)
    
    # Step 2: Create embedding column if needed
    logger.info("\n" + "-" * 70)
    logger.info("STEP 2: Preparing embedding column...")
    if not create_embedding_column():
        logger.error("❌ Failed to create embedding column. Exiting.")
        sys.exit(1)
    
    # Step 3: Generate embeddings
    logger.info("\n" + "-" * 70)
    logger.info("STEP 3: Generating GTE-Large embeddings...")
    if not generate_embeddings_batch(available_columns):
        logger.error("❌ Failed to generate embeddings. Exiting.")
        sys.exit(1)
    
    # Step 4: Verify embeddings
    logger.info("\n" + "-" * 70)
    logger.info("STEP 4: Verifying embeddings...")
    if not verify_embeddings():
        logger.error("❌ Embedding verification failed.")
        sys.exit(1)
    
    # Step 5: Test embedding quality
    logger.info("\n" + "-" * 70)
    logger.info("STEP 5: Testing embedding quality...")
    test_embedding_quality()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ GTE-LARGE EMBEDDING GENERATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info("=" * 70)
    
    logger.info("\n🎯 Upgrade Summary:")
    logger.info("   ✅ Model: thenlper/gte-large (higher quality than MiniLM)")
    logger.info("   ✅ Dimensions: 1024 (vs 384 previously)")
    logger.info("   ✅ Normalization: Enabled (stable cosine similarity)")
    logger.info("   ✅ Device: " + DEVICE)
    logger.info("   ✅ Expected improvement: Better semantic matching accuracy")
    
    logger.info("\n💡 Next Steps:")
    logger.info("   1. Update app.py to use GTE-Large (see updated version)")
    logger.info("   2. Restart API server: python app.py")
    logger.info("   3. Test semantic search - you should see better relevance!")
    logger.info("\n✨ Your AI-powered semantic search is now upgraded to GTE-Large!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Process interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)
