import sqlite3
import os
from datetime import datetime
import json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'evaluation_logs.db')

def get_connection():
    """Create and return a database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for storing inference results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            image_path TEXT NOT NULL,
            ground_truth TEXT,
            generated_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for storing evaluation metrics (BLEU, ROUGE, etc.) linked to inferences
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            details TEXT,  -- JSON string for extended metric details
            FOREIGN KEY (inference_id) REFERENCES inferences (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def log_inference(model_version, image_path, generated_text, ground_truth=None):
    """Log a single inference output."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO inferences (model_version, image_path, ground_truth, generated_text, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (model_version, image_path, ground_truth, generated_text, datetime.now()))
    
    inference_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inference_id

def log_metric(inference_id, metric_name, metric_value, details=None):
    """Log an evaluation metric for a specific inference."""
    conn = get_connection()
    cursor = conn.cursor()
    
    details_str = json.dumps(details) if details else None
    
    cursor.execute('''
        INSERT INTO metrics (inference_id, metric_name, metric_value, details)
        VALUES (?, ?, ?, ?)
    ''', (inference_id, metric_name, metric_value, details_str))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
