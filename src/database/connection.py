"""
Database connection for fraud detection system.
PostgreSQL 18 on D: drive, port 5433.
Credentials loaded from .env file — NEVER commit .env to git.
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database config — ALL values from environment variables
# Create a .env file in project root with:
#   FRAUD_DB_HOST=localhost
#   FRAUD_DB_PORT=5433
#   FRAUD_DB_NAME=fraud_detection
#   FRAUD_DB_USER=postgres
#   FRAUD_DB_PASSWORD=your_password

DB_CONFIG = {
    'host': os.getenv('FRAUD_DB_HOST', 'localhost'),
    'port': os.getenv('FRAUD_DB_PORT', '5433'),
    'database': os.getenv('FRAUD_DB_NAME', 'fraud_detection'),
    'user': os.getenv('FRAUD_DB_USER', 'postgres'),
    'password': os.getenv('FRAUD_DB_PASSWORD'),  # Required — set in .env
}

def get_connection_string():
    if not DB_CONFIG['password']:
        raise RuntimeError(
            "FRAUD_DB_PASSWORD not set. "
            "Create a .env file or set the environment variable."
        )
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

engine = create_engine(
    get_connection_string(),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine)

@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def test_connection():
    """Quick connection test."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False