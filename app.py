from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import whisper
import requests
import sys
import json
import socket
import re
from collections import Counter

# Try to import NLTK, but provide fallbacks if not available
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    
    # Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available. Using simplified text processing.")
    NLTK_AVAILABLE = False
    
    # Define fallback functions
    def sent_tokenize(text):
        # Simple sentence tokenization based on punctuation
        return re.split(r'(?<=[.!?]\s)', text)
    
    # Mock stopwords with a small set of common English stopwords
    class MockStopwords:
        def words(self, language):
            return ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    
    stopwords = MockStopwords()


def process_transcript(text):
    """Process transcript for improved readability and key point highlighting."""
    if not text:
        return ""
    
    # Split into paragraphs based on natural breaks (pauses, speakers, etc.)
    paragraphs = []
    current_paragraph = []
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        # Create a new paragraph after every 2-3 sentences or if sentence ends with a question mark
        if len(current_paragraph) >= 3 or sentence.endswith('?'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add any remaining sentences as a paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Identify key points and important sentences
    key_points = identify_key_points(text, sentences)
    
    # Format the transcript with HTML
    formatted_transcript = "<div class='transcript-content'>"
    
    # Add paragraphs with key points highlighted
    for paragraph in paragraphs:
        paragraph_html = paragraph
        
        # Highlight key sentences in the paragraph
        for key_sentence in key_points:
            if key_sentence in paragraph:
                highlighted = f"<span class='highlight'>{key_sentence}</span>"
                paragraph_html = paragraph_html.replace(key_sentence, highlighted)
        
        formatted_transcript += f"<p>{paragraph_html}</p>"
    
    # Add key points summary at the top
    if key_points:
        formatted_transcript = "<div class='key-points-section'><h3>Key Points</h3><ul>" + \
                            "".join([f"<li>{point}</li>" for point in key_points[:5]]) + \
                            "</ul></div>" + formatted_transcript
    
    formatted_transcript += "</div>"
    return formatted_transcript


def identify_key_points(text, sentences):
    """Identify key points in the transcript based on important keywords and phrases."""
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Common phrases that might indicate important points
    important_phrases = [
        "key point", "important", "remember", "note that", "crucial", 
        "essential", "significant", "highlight", "takeaway", "conclusion",
        "in summary", "to summarize", "main point", "priority", "focus on",
        "keep in mind", "don't forget", "remember that", "critical"
    ]
    
    # Extract all words and count their frequency
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    
    # Find important sentences based on keyword frequency and important phrases
    scored_sentences = []
    for sentence in sentences:
        score = 0
        # Score based on word frequency
        for word in re.findall(r'\b\w+\b', sentence.lower()):
            if word in word_freq:
                score += word_freq[word]
        
        # Boost score if sentence contains important phrases
        for phrase in important_phrases:
            if phrase in sentence.lower():
                score += 10
        
        # Boost score for sentences with numbers (often important statistics/facts)
        if re.search(r'\d+', sentence):
            score += 5
            
        # Boost score for shorter sentences (often more impactful)
        if len(sentence.split()) < 15:
            score += 3
            
        scored_sentences.append((sentence, score))
    
    # Sort sentences by score and get top ones
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    key_points = [sentence for sentence, score in scored_sentences[:8]]
    
    return key_points


def diagnose_lm_studio_connection():
    diagnostics = {
        "network_checks": {},
        "endpoint_checks": {}
    }
    test_endpoints = ["localhost", "127.0.0.1", "192.168.0.101"]
    test_ports = [1234, 11434, 8000]
    for endpoint in test_endpoints:
        for port in test_ports:
            key = f"{endpoint}:{port}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((endpoint, port))
                diagnostics["network_checks"][key] = "Open" if result == 0 else "Closed"
                sock.close()
            except Exception as e:
                diagnostics["network_checks"][key] = f"Error: {str(e)}"

    test_urls = [
        "http://localhost:1234",
        "http://localhost:1234/v1/chat/completions",
        "http://localhost:1234/chat",
        "http://127.0.0.1:1234",
        "http://192.168.0.101:1234"
    ]
    for url in test_urls:
        try:
            response = requests.get(url, timeout=3)
            diagnostics["endpoint_checks"][url] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:200]
            }
        except requests.ConnectionError:
            diagnostics["endpoint_checks"][url] = "Connection Error"
        except requests.Timeout:
            diagnostics["endpoint_checks"][url] = "Timeout"
        except Exception as e:
            diagnostics["endpoint_checks"][url] = f"Error: {str(e)}"
    return diagnostics


app = Flask(__name__)
app.secret_key = "my-dev-secret-key-123"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base")


@app.route("/")
def index():
    return render_template(
        "index.html",
        transcript=session.get("transcript", ""),
        action_points=session.get("action_points", "")
    )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    mode = request.form.get("mode", "script")
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Process the transcription
    result = model.transcribe(filepath)
    full_transcript = result["text"]
    
    # Process transcript for improved readability
    processed_transcript = process_transcript(full_transcript)
    
    # Store in session
    session["transcript"] = processed_transcript
    session["raw_transcript"] = full_transcript
    session["mode"] = mode
    
    # Return success response
    return jsonify({
        "success": True,
        "message": "Transcription complete",
        "redirect": url_for("chat")
    })


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.args.get('diagnose') == 'true':
        diagnostics = diagnose_lm_studio_connection()
        return jsonify(diagnostics)

    if request.method == 'POST':
        user_question = request.json.get('user_input', '')
        # Use the raw transcript for question answering to avoid HTML formatting issues
        transcript = session.get("raw_transcript", session.get("transcript", ''))
        
        # Strip HTML tags if present in the transcript
        transcript = re.sub(r'<.*?>', '', transcript)

        prompt = f"""
You are an AI assistant helping to answer questions based ONLY on the following video transcript.

Transcript:
{transcript}

User Question:
{user_question}

Answer only using the transcript. If the transcript does not contain the answer, reply: 'Sorry, I couldn't find that in the video.'
Make your answer concise and highlight any important information or numbers.
"""

        print(f"Received prompt: {user_question}", file=sys.stderr)

        lmstudio_url = "http://192.168.0.101:1234/v1/completions"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7
        }

        try:
            print(f"Attempting to connect to: {lmstudio_url}", file=sys.stderr)
            print(f"Payload: {json.dumps(payload)}", file=sys.stderr)

            response = requests.post(lmstudio_url, json=payload, headers=headers, timeout=50)
            print(f"Response Status: {response.status_code}", file=sys.stderr)
            print(f"Response Headers: {response.headers}", file=sys.stderr)
            print(f"Response Text: {response.text[:500]}", file=sys.stderr)

            if response.status_code == 200:
                try:
                    response_json = response.json()
                    answer = response_json.get("choices", [{}])[0].get("text", "").strip()
                    if not answer:
                        answer = "[No response content from model]"
                    return jsonify({"response": answer})
                except json.JSONDecodeError as je:
                    print(f"JSON Decode Error: {je}", file=sys.stderr)
                    return jsonify({"response": f"[JSON Parsing Error] {str(je)}"})

            else:
                return jsonify({"response": f"[LM Studio Error] {response.status_code}: {response.text[:200]}"})

        except requests.ConnectionError as ce:
            print(f"Connection Error: {ce}", file=sys.stderr)
            return jsonify({"response": f"[Connection Error] Unable to reach LM Studio server. Check server status."})
        except requests.Timeout as te:
            print(f"Connection Timeout: {te}", file=sys.stderr)
            return jsonify({"response": f"[Timeout Error] LM Studio server took too long to respond."})
        except requests.RequestException as e:
            print(f"Request Exception: {e}", file=sys.stderr)
            return jsonify({"response": f"[Request Exception] {str(e)}"})

    return render_template("chat.html")


if __name__ == "__main__":
    app.run(debug=True)
