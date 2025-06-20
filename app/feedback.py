import sqlite3
import os

DB_FILE = 'feedback.db'

# Ensure DB exists
def initialize_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comment TEXT,
                prediction INTEGER,
                user_corrected INTEGER
            )
        ''')
        conn.commit()

# Save user feedback
def save_feedback(comment, prediction, user_corrected):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (comment, prediction, user_corrected) VALUES (?, ?, ?)",
            (comment, prediction, user_corrected)
        )
        conn.commit()

# Retrieve feedback
def get_feedback():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback")
        return cursor.fetchall()

# Run this once at startup
initialize_db()
