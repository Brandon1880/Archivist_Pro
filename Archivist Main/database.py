import sqlite3
from datetime import datetime

DB_FILE = "scan_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (filename TEXT PRIMARY KEY, date_scanned TEXT, status TEXT)''')
    conn.commit()
    conn.close()

def check_history(filename):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT status FROM history WHERE filename=?", (filename,))
    result = c.fetchone()
    conn.close()
    return result is not None

def mark_scanned(filename):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT OR REPLACE INTO history VALUES (?, ?, ?)", 
              (filename, timestamp, "COMPLETED"))
    conn.commit()
    conn.close()