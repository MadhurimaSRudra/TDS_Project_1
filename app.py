import os
import requests
import json
import subprocess
import sqlite3
import duckdb
import markdown
import pandas as pd
import whisper
from bs4 import BeautifulSoup
from PIL import Image
from flask import Flask, request, jsonify

# === SECURITY CHECKS (B1 & B2) ===
def enforce_security(filepath, allow_deletion=False):
    """Ensures file access is within /data and prevents deletion unless explicitly allowed."""
    if not filepath.startswith('/data'):
        raise PermissionError(f"Access outside /data is not allowed: {filepath}")
    if not allow_deletion and os.path.exists(filepath):
        raise PermissionError(f"Deletion/modification of existing files is not allowed: {filepath}")
    return True

# === B3: Fetch Data from an API ===
def fetch_and_save_data(url, save_path):
    enforce_security(save_path)
    
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data. HTTP {response.status_code}: {response.text}")
    
    data = response.json() if 'application/json' in response.headers.get('Content-Type', '') else response.text
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4) if isinstance(data, dict) else file.write(data)

# === B4: Clone a Git Repo and Make a Commit ===
def clone_git_repo(repo_url, commit_message):
    repo_path = "/data/repo"
    enforce_security(repo_path)

    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    
    subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_message], check=True)

# === B5: Execute SQL Query and Save Results ===
def execute_sql_query(db_path, query, output_filename):
    enforce_security(db_path)
    enforce_security(output_filename)
    
    conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(query)
        result = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        output_data = [dict(zip(columns, row)) for row in result]
    except Exception as e:
        raise RuntimeError(f"SQL execution failed: {str(e)}")
    finally:
        conn.close()

    with open(output_filename, 'w') as file:
        json.dump(output_data, file, indent=4)

# === B6: Web Scraping ===
def scrape_website(url, output_filename):
    enforce_security(output_filename)
    
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to scrape data. HTTP {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    extracted_text = soup.get_text()

    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(extracted_text)

# === B7: Image Processing (Resizing & Saving) ===
def process_image(image_path, output_path, resize=None):
    enforce_security(image_path)
    enforce_security(output_path)

    try:
        img = Image.open(image_path)
        if resize:
            img = img.resize(resize)
        img.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

# === B8: Audio Transcription ===
def transcribe_audio(audio_path, output_path):
    enforce_security(audio_path)
    enforce_security(output_path)

    model = whisper.load_model("small")
    result = model.transcribe(audio_path)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(result["text"])

# === B9: Markdown to HTML Conversion ===
def convert_md_to_html(md_path, output_path):
    enforce_security(md_path)
    enforce_security(output_path)

    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            html = markdown.markdown(file.read())
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(html)
    except Exception as e:
        raise RuntimeError(f"Markdown conversion failed: {str(e)}")

# === B10: API Endpoint for CSV Filtering ===
app = Flask(__name__)

@app.route('/filter_csv', methods=['POST'])
def filter_csv():
    data = request.json
    csv_path, filter_column, filter_value = data.get('csv_path'), data.get('filter_column'), data.get('filter_value')

    enforce_security(csv_path)

    try:
        df = pd.read_csv(csv_path)
        if filter_column not in df.columns:
            return jsonify({"error": f"Column '{filter_column}' does not exist"}), 400

        filtered = df[df[filter_column] == filter_value]
        return jsonify(filtered.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
