import sqlite3
import json
import os
from pathlib import Path

_DATA_DIR = Path('data') / 'clean-data'

conn = sqlite3.connect("cartoons.sqlite")
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE Cartoons 
             (title text, keywords text, caption text, image_url text, author text, date text)''')

for filename in os.listdir(_DATA_DIR):
    cid, ext = os.path.splitext(filename)

    if ext != '.json':
        continue

    with open(_DATA_DIR / filename, 'r') as f:
        metadata = json.load(f)
        values = (metadata["title"], metadata["keywords"], metadata["caption"],
                  metadata["image_url"], metadata["author"], metadata["date"])
        c.execute("INSERT INTO Cartoons VALUES (?, ?, ?, ?, ?, ?)", values)

# Save changes and exit
conn.commit()
conn.close()
