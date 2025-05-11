from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem, Image as RLImage
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import HexColor, Color, white, black, CMYKColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String, Image
from reportlab.graphics import renderPDF
import tempfile
from datetime import datetime, timedelta
import os
from PIL import Image as PILImage
import math
import qrcode
import re
import requests
from pytube import YouTube
from io import BytesIO, StringIO
from typing import List, Dict
import re
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
        "http://192.168.1.12:1234"
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

# Configure logging
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Favicon handler
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


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
    
    # Process transcript for improved readability and get summary
    sentences = sent_tokenize(full_transcript)
    key_points = identify_key_points(full_transcript, sentences)
    summary = generate_summary(full_transcript, key_points)
    processed_transcript = process_transcript(full_transcript)
    
    # Store in session
    session["transcript"] = processed_transcript
    session["raw_transcript"] = full_transcript
    session["mode"] = mode
    session["timestamps"] = result.get("segments", [])
    session["summary"] = summary
    
    # Return success response
    return jsonify({
        "success": True,
        "message": "Transcription complete",
        "redirect": url_for("chat", summary=summary)
    })


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == 'GET':
        summary = session.get('summary', '')
        return render_template("chat.html", summary=summary)
    if request.args.get('diagnose') == 'true':
        diagnostics = diagnose_lm_studio_connection()
        return jsonify(diagnostics)

    if request.method == 'POST':
        # Validate JSON input
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        user_question = request.json.get('user_input', '')
        if not user_question:
            return jsonify({"error": "User input is required"}), 400

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

        lmstudio_url = "http://192.168.1.12:1234/v1/completions"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            print(f"Attempting to connect to: {lmstudio_url}", file=sys.stderr)
            print(f"Payload: {json.dumps(payload)}", file=sys.stderr)

            response = requests.post(lmstudio_url, json=payload, headers=headers, timeout=100)
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
                    return jsonify({"error": "JSON parsing failed", "details": str(je)}), 500

            else:
                return jsonify({"error": f"LM Studio Error {response.status_code}", "details": response.text[:200]}), response.status_code

        except requests.ConnectionError as ce:
            print(f"Connection Error: {ce}", file=sys.stderr)
            return jsonify({"error": "Connection failed", "details": str(ce)}), 500
        except requests.Timeout as te:
            print(f"Connection Timeout: {te}", file=sys.stderr)
            return jsonify({"error": "Request timed out", "details": str(te)}), 504
        except requests.RequestException as e:
            print(f"Request Exception: {e}", file=sys.stderr)
            return jsonify({"error": "Request failed", "details": str(e)}), 500

    return render_template("chat.html")


class TranscriptPDF:
    def is_youtube_url(self, url):
        """Check if the given URL is a YouTube URL and return video ID"""
        if not url:
            return None
        youtube_regex = r'(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^"&?/ ]{11})'
        match = re.search(youtube_regex, url)
        return match.group(1) if match else None
        
    def get_youtube_thumbnail(self, video_id):
        """Get YouTube video thumbnail"""
        try:
            yt = YouTube(f'https://youtube.com/watch?v={video_id}')
            return yt.thumbnail_url
        except Exception as e:
            print(f"Error getting YouTube thumbnail: {e}")
            return None

    def create_qr_code(self, url):
        """Create a QR code for the given URL"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Create QR code image
        qr_img = qr.make_image(fill_color=str(self.secondary_color), back_color="white")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            qr_img.save(tmp.name)
            return tmp.name

    def __init__(self, filename):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self.elements = []
        self.page_width, self.page_height = letter
        self.bookmarks = []
        
        # Define colors
        self.primary_color = HexColor('#0074d9')
        self.secondary_color = HexColor('#001f3f')
        self.accent_color = HexColor('#7FDBFF')
        self.text_color = HexColor('#2C3E50')
        self.gray_color = HexColor('#95A5A6')
        self.light_color = HexColor('#F8F9FA')
        
        # Cover page gradient colors
        self.gradient_start = CMYKColor(0.85, 0.4, 0, 0.2)
        self.gradient_end = CMYKColor(0.6, 0.2, 0, 0)
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=self.primary_color,
            spaceAfter=30,
            alignment=TA_CENTER,
            leading=32
        )
        
        self.heading_style = ParagraphStyle(
            'Heading',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=self.secondary_color,
            spaceBefore=25,
            spaceAfter=15,
            alignment=TA_LEFT,
            leading=22,
            borderWidth=1,
            borderColor=self.accent_color,
            borderPadding=10,
            backColor=HexColor('#F8F9FA')
        )
        
        self.body_style = ParagraphStyle(
            'Body',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=18,
            textColor=self.text_color,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6
        )
        
        self.timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.accent_color,
            alignment=TA_RIGHT,
            spaceBefore=4,
            spaceAfter=4
        )
        
        self.summary_style = ParagraphStyle(
            'Summary',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=18,
            textColor=self.text_color,
            alignment=TA_LEFT,
            leftIndent=20,
            spaceBefore=4,
            spaceAfter=4,
            bulletIndent=10,
            borderWidth=0.5,
            borderColor=self.accent_color,
            borderPadding=10,
            borderRadius=5
        )
        self.meta_style = ParagraphStyle(
            'MetaInfo',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=self.gray_color,
            spaceAfter=6
        )
        self.footer_style = ParagraphStyle(
            'Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=self.gray_color,
            alignment=TA_RIGHT
        )
    
    def header(self, canvas, doc):
        # Save state
        canvas.saveState()

        # Add header with gradient background
        header_height = 50
        canvas.setFillColor(self.primary_color)
        canvas.rect(0, doc.pagesize[1] - header_height, doc.pagesize[0], header_height, fill=True)
        
        # Add header text
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 12)
        canvas.drawString(72, doc.pagesize[1] - 30, "Video Transcript")
        
        # Add a line under the header
        canvas.setStrokeColor(self.accent_color)
        canvas.setLineWidth(2)
        canvas.line(72, doc.pagesize[1] - header_height + 10, doc.pagesize[0] - 72, doc.pagesize[1] - header_height + 10)

        # Add footer with metadata
        footer_y = 30
        canvas.setFillColor(self.gray_color)
        canvas.setFont('Helvetica', 8)
        footer_text = f'Generated by Sweya.AI Video Transcriber | Page {doc.page} | {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        canvas.drawString(72, footer_y, footer_text)

        # Add a line above the footer
        canvas.setStrokeColor(self.accent_color)
        canvas.setLineWidth(0.5)
        canvas.line(72, footer_y + 10, doc.pagesize[0] - 72, footer_y + 10)

        # Restore state
        canvas.restoreState()

        # Add bookmarks for the current page
        for bookmark in self.bookmarks:
            if bookmark['page'] == doc.page:
                key = f"{bookmark['level']}.{bookmark['text']}"
                canvas.bookmarkPage(key)
                canvas.addOutlineEntry(bookmark['text'], key, bookmark['level'])
        
        canvas.restoreState()
    
    def create_toc_entry(self, text, level=0):
        """Create a table of contents entry with the given text and level"""
        style = ParagraphStyle(
            f'TOC_Level{level}',
            parent=self.styles['Normal'],
            fontSize=12 - level,
            leftIndent=20 * level,
            spaceBefore=5,
            spaceAfter=5
        )
        return Paragraph(text, style)

    def create_cover_drawing(self, video_info):
        """Create an elegant cover page with geometric design"""
        # Use margins from document template
        width = 480  # page width minus margins
        height = 675  # page height minus margins
        d = Drawing(width, height)
        
        # Background gradient
        num_stripes = 40
        stripe_height = height / num_stripes
        for i in range(num_stripes):
            y = i * stripe_height
            progress = i / num_stripes
            c = CMYKColor(
                self.gradient_start.cyan * (1 - progress) + self.gradient_end.cyan * progress,
                self.gradient_start.magenta * (1 - progress) + self.gradient_end.magenta * progress,
                self.gradient_start.yellow * (1 - progress) + self.gradient_end.yellow * progress,
                self.gradient_start.black * (1 - progress) + self.gradient_end.black * progress
            )
            d.add(Rect(0, y, width, stripe_height, fillColor=c, strokeColor=None))
        
        # Decorative circles
        circle_color = white
        circle_size = 80
        spacing = 100
        for x in range(0, int(width) + spacing, spacing):
            for y in range(0, int(height) + spacing, spacing):
                d.add(Circle(x, y, circle_size/2, fillColor=None, strokeColor=circle_color, strokeWidth=0.5))
        
        # Video icon
        icon_size = 120
        icon_x = width/2
        icon_y = height * 0.6
        d.add(Circle(icon_x, icon_y, icon_size/2, fillColor=white, strokeColor=None))
        d.add(Circle(icon_x, icon_y, icon_size/2-10, fillColor=self.primary_color, strokeColor=None))
        
        # Play triangle
        triangle_size = 40
        d.add(String(icon_x-triangle_size/4, icon_y-triangle_size/4, '▶', fontSize=triangle_size, fillColor=white))
        
        # Create cover page
        cover = renderPDF.GraphicsFlowable(d)
        cover.width = width
        cover.height = height
        self.elements.append(cover)
        
        # Add QR code and thumbnail if it's a YouTube video
        video_url = video_info.get('url')
        video_id = self.is_youtube_url(video_url)
        if video_id:
            # Get video info including thumbnail
            try:
                yt = YouTube(f'https://youtube.com/watch?v={video_id}')
                youtube_info = {
                    'title': yt.title,
                    'author': yt.author,
                    'views': yt.views,
                    'duration': self.format_duration(yt.length),
                    'thumbnail': yt.thumbnail_url
                }
                if youtube_info:
                    qr_path = self.create_qr_code(video_url)
                    qr_size = 100
                    qr_x = width - qr_size - 40
                    qr_y = 40
                    
                    # Add thumbnail as background
                    thumb_x = width/2 - 200
                    thumb_y = height * 0.25
                    thumb_width = 400
                    thumb_height = 225  # 16:9 aspect ratio
                    
                    # Add thumbnail background and border
                    d.add(Rect(thumb_x-5, thumb_y-5, thumb_width+10, thumb_height+10, fillColor=white, strokeColor=None))
                    d.add(Rect(thumb_x-3, thumb_y-3, thumb_width+6, thumb_height+6, fillColor=None, strokeColor=self.accent_color, strokeWidth=1))
                    
                    # Add thumbnail
                    thumb_img = Image(thumb_x, thumb_y, thumb_width, thumb_height, youtube_info['thumbnail'])
                    d.add(thumb_img)
                    
                    # Add video title
                    title_style = ParagraphStyle(
                        'VideoTitle',
                        fontName='Helvetica-Bold',
                        fontSize=14,
                        textColor=white,
                        alignment=TA_CENTER
                    )
                    title = Paragraph(youtube_info['title'][:50] + '...' if len(youtube_info['title']) > 50 else youtube_info['title'], title_style)
                    title.wrapOn(d, thumb_width, 50)
                    title.drawOn(d, thumb_x, thumb_y + thumb_height + 10)
                    
                    # Add QR code background and border
                    d.add(Rect(qr_x-10, qr_y-10, qr_size+20, qr_size+20, fillColor=white, strokeColor=None))
                    d.add(Rect(qr_x-8, qr_y-8, qr_size+16, qr_size+16, fillColor=None, strokeColor=self.accent_color, strokeWidth=1))
                    
                    # Add QR code image
                    qr_img = Image(qr_x, qr_y, qr_size, qr_size, qr_path)
                    d.add(qr_img)
                    
                    # Add "Scan for Video" text
                    d.add(String(qr_x + qr_size/2 - 30, qr_y-25, 'Scan for Video', fontSize=10, fillColor=white))
                    
                    # Clean up temporary files
                    os.unlink(qr_path)
                    os.unlink(youtube_info['thumbnail'])
                    
                    # Update video info
                    video_info.update(youtube_info)
            except Exception as e:
                print(f"Error processing YouTube video: {e}")
                # Continue without YouTube info

        
        return d

    def format_duration(self, seconds):
        """Format duration in seconds to HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
        return f"{int(minutes):02d}:{int(seconds):02d}"

    def create_pdf(self, transcript, summary, timestamps=None, video_info=None, video_path=None):
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=letter,
            rightMargin=60,
            leftMargin=60,
            topMargin=60,
            bottomMargin=45
        )
        
        # Get video metadata
        if not video_info:
            video_info = {}
        if timestamps:
            duration = timestamps[-1]['end'] - timestamps[0]['start']
            video_info.update({
                'Duration': self.format_duration(duration),
                'Segments': len(timestamps),
                'Words': len(transcript.split()),
                'Generated': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        
        # Create cover page
        self.create_cover_drawing(video_info)
        
        # Add title with icon
        self.elements.append(Spacer(1, 40))
        title_with_icon = '''<para alignment="center">
            <font color="#{0}" size="28">📽 {1}</font>
        </para>'''.format(
            self.primary_color.hexval()[2:],
            video_info.get('title', 'Video Transcript')
        )
        self.elements.append(Paragraph(title_with_icon, self.title_style))
        self.elements.append(Spacer(1, 20))
        
        # Add video metadata with modern card styling
        meta_style = ParagraphStyle(
            'MetaCard',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=self.text_color,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=4,
            borderWidth=0.5,
            borderColor=HexColor('#E9ECEF'),
            borderPadding=15,
            borderRadius=8,
            backColor=HexColor('#F8F9FA')
        )

        meta_data = [
            ['📹 Title', video_info.get('title', 'N/A')],
            ['👤 Author', video_info.get('author', 'N/A')],
            ['⏱ Duration', video_info.get('duration', 'N/A')],
            ['👁 Views', video_info.get('views', 'N/A')],
            ['📝 Segments', len(timestamps) if timestamps else 'N/A'],
            ['📊 Total Words', len(transcript.split()) if transcript else 'N/A'],
            ['🕒 Generated', datetime.now().strftime('%Y-%m-%d %H:%M')]
        ]
        
        # Create metadata cards
        for label, value in meta_data:
            meta_text = f'''<para backColor="#F8F9FA" borderColor="#E9ECEF">
                <b>{label}:</b> {value}
            </para>'''
            self.elements.append(Paragraph(meta_text, meta_style))
            self.elements.append(Spacer(1, 5))
        
        self.elements.append(PageBreak())
        
        # Meta information table
        meta_data = [
            ['Generated Date:', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Document Type:', 'Video Transcript'],
            ['Length:', f'{len(transcript.split())} words']
        ]
        meta_table = Table(meta_data, colWidths=[100, 300])
        meta_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.gray),
            ('FONT', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        self.elements.append(meta_table)
        self.elements.append(Spacer(1, 30))

        # Table of Contents
        self.elements.append(Paragraph('Table of Contents', self.heading_style))
        
        # Add sections to table of contents
        self.elements.append(self.create_toc_entry('Video Information', 0))
        self.elements.append(self.create_toc_entry('Key Points', 0))
        self.elements.append(self.create_toc_entry('Transcript', 0))
        self.elements.append(Spacer(1, 20))
        
        return video_info
    
    def format_duration(self, seconds):
        """Format duration in seconds to HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
        return f"{int(minutes):02d}:{int(seconds):02d}"

    def create_pdf(self, transcript, summary, timestamps=None, video_info=None, video_path=None):
        """Create a PDF document with the transcript and video information"""
        try:
            print("Starting create_pdf...")
            # Initialize document with all margins
            doc = SimpleDocTemplate(
                self.filename,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            self.elements = []  # Reset elements
            print("Document initialized")
            
            # Add title
            self.elements.append(Paragraph('Video Transcript', self.title_style))
            self.elements.append(Spacer(1, 30))
            
            # Add video information
            if video_info:
                try:
                    print("Adding video information...")
                    self.elements.append(Paragraph('Video Information', self.heading_style))
                    
                    info_items = [
                        ('Title', video_info.get('title', 'N/A')),
                        ('Duration', video_info.get('duration', 'N/A'))
                    ]
                    
                    # Create metadata table
                    table_data = []
                    for label, value in info_items:
                        if value and value != 'N/A':
                            table_data.append([label, str(value)])
                    
                    if table_data:
                        meta_table = Table(table_data, colWidths=[100, 300])
                        meta_table.setStyle(TableStyle([
                            ('TEXTCOLOR', (0, 0), (-1, -1), self.text_color),
                            ('FONT', (0, 0), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 0), (-1, -1), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                            ('TOPPADDING', (0, 0), (-1, -1), 6)
                        ]))
                        self.elements.append(meta_table)
                        self.elements.append(Spacer(1, 20))
                        print("Video information added")
                except Exception as e:
                    print(f"Error adding video information: {e}")
            
            # Add summary section
            if summary:
                try:
                    print("Adding summary section...")
                    self.elements.append(Paragraph('Summary', self.heading_style))
                    self.elements.append(Paragraph(summary, self.body_style))
                    self.elements.append(Spacer(1, 20))
                    print("Summary section added")
                except Exception as e:
                    print(f"Error adding summary section: {e}")
            
            # Add transcript
            if transcript:
                try:
                    print("Adding transcript...")
                    self.elements.append(Paragraph('Transcript', self.heading_style))
                    if timestamps:
                        # Add transcript with timestamps
                        for segment in timestamps:
                            timestamp = segment.get('timestamp', '')
                            text = segment.get('text', '').strip()
                            if text:
                                # Add timestamp if available
                                if timestamp:
                                    self.elements.append(Paragraph(timestamp, self.timestamp_style))
                                # Add text
                                self.elements.append(Paragraph(text, self.body_style))
                                self.elements.append(Spacer(1, 8))
                    else:
                        # Add transcript as plain text
                        self.elements.append(Paragraph(transcript, self.body_style))
                    print("Transcript added")
                except Exception as e:
                    print(f"Error adding transcript: {e}")
            
            # Build PDF
            print("Building PDF...")
            doc.build(self.elements)
            print("PDF built successfully")
            return self.filename
            
        except Exception as e:
            print(f"Error in create_pdf: {e}")
            import traceback
            traceback.print_exc()
            raise

def clean_text(text):
    """Clean text by removing HTML tags and extra whitespace"""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Fix bullet points
    text = text.replace('•', '* ')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_key_points(transcript):
    """Extract key points from the transcript"""
    # Clean transcript first
    transcript = clean_text(transcript)
    
    try:
        # Try using NLTK for better sentence tokenization
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(transcript)
    except:
        # Fallback to simple split
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    
    # Get important sentences (first few and any with important keywords)
    key_points = []
    keywords = ['important', 'key', 'main', 'significant', 'crucial', 'essential', 'primary']
    
    # Add first sentence as it often contains the main point
    if sentences:
        key_points.append(sentences[0])
    
    # Add sentences with keywords
    for sentence in sentences[1:]:
        if any(keyword in sentence.lower() for keyword in keywords):
            key_points.append(sentence)
    
    # Add a few more sentences if we don't have enough key points
    if len(key_points) < 3 and len(sentences) > 1:
        for sentence in sentences[1:4]:
            if sentence not in key_points:
                key_points.append(sentence)
    
    return key_points

def format_summary(transcript):
    """Generate a formatted summary from the transcript"""
    if not transcript:
        return ''
    
    # Extract key points
    key_points = extract_key_points(transcript)
    
    # Format summary
    if key_points:
        summary = '\n'.join(f'• {point}' for point in key_points)
    else:
        # Fallback to simple text truncation
        summary = transcript[:500] + '...' if len(transcript) > 500 else transcript
    
    return summary

def generate_pdf(transcript, summary=None, timestamps=None, video_info=None, video_path=None):
    """Generate a PDF file with the transcript and video information"""
    try:
        print("Starting PDF generation...")
        import os
        from datetime import datetime
        import tempfile
        
        # Create temporary file with a unique name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'transcript_{timestamp}.pdf'
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, filename)
        print(f"PDF will be saved to: {pdf_path}")
        
        # Clean transcript and summary
        transcript = clean_text(transcript) if transcript else ''
        print(f"Transcript length: {len(transcript)}")
        
        # Always generate a fresh summary to ensure we have key points
        if transcript:
            summary = format_summary(transcript)
            print(f"Summary length: {len(summary)}")
        
        # Create PDF instance
        pdf = TranscriptPDF(pdf_path)
        
        # Generate PDF and return the path
        print("Creating PDF...")
        pdf.create_pdf(transcript, summary, timestamps, video_info, video_path)
        
        if os.path.exists(pdf_path):
            print(f"PDF generated successfully at {pdf_path}")
            return pdf_path
        else:
            print("PDF file was not created")
            raise Exception("PDF file was not created")
            
    except Exception as e:
        print(f"Error in generate_pdf: {e}")
        raise

@app.route('/download/<format>')
def download_transcript(format):
    try:
        # Get transcript and metadata from session
        transcript = session.get('transcript')
        if not transcript:
            print("No transcript found in session")
            return jsonify({'error': 'No transcript found in session'}), 404
            
        print(f"Got transcript from session, length: {len(transcript)}")
        summary = session.get('summary', '')
        print(f"Got summary, length: {len(summary)}")
        timestamps = session.get('timestamps', [])
        print(f"Got timestamps, count: {len(timestamps)}")
        video_info = session.get('video_info', {})
        print(f"Got video info: {video_info}")
        video_path = session.get('video_path', '')
        print(f"Got video path: {video_path}")
        
        if format == 'pdf':
            try:
                print("Starting PDF generation...")
                # Generate PDF
                pdf_path = generate_pdf(transcript, summary, timestamps, video_info, video_path)
                print(f"PDF generated at: {pdf_path}")
                
                if not os.path.exists(pdf_path):
                    print(f"PDF file not found at: {pdf_path}")
                    return jsonify({'error': 'PDF file not found after generation'}), 500
                
                print(f"PDF file size: {os.path.getsize(pdf_path)} bytes")
                
                # Return file with Content-Disposition: inline for browser preview
                try:
                    response = send_file(
                        pdf_path,
                        mimetype='application/pdf',
                        download_name='transcript.pdf',
                        as_attachment=False,
                        conditional=True,
                        etag=True,
                        last_modified=datetime.now()
                    )
                    print("PDF file sent successfully")
                    return response
                except Exception as e:
                    print(f"Error sending PDF file: {e}")
                    return jsonify({'error': f'Failed to send PDF: {str(e)}'}), 500
            except Exception as e:
                print(f"Error in PDF generation: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500
        
        elif format == 'txt':
            # Return plain text
            return Response(
                transcript,
                mimetype='text/plain',
                headers={'Content-Disposition': 'attachment;filename=transcript.txt'}
            )
        
        else:
            return jsonify({'error': 'Invalid format requested'}), 400
            
    except Exception as e:
        print(f"Error in download_transcript: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def search_transcript_segments(query: str, segments: List[Dict]) -> List[Dict]:
    """Search through transcript segments for matching text."""
    results = []
    query = query.lower()
    
    # Split query into words for better matching
    query_words = query.split()
    
    for segment in segments:
        text = segment.get('text', '').lower()
        timestamp = segment.get('start', 0)
        
        # Convert timestamp to seconds if it's in HH:MM:SS format
        if isinstance(timestamp, str):
            try:
                parts = timestamp.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    timestamp = h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    timestamp = m * 60 + s
            except:
                timestamp = 0
        
        # Check if all query words are in the text
        if all(word in text for word in query_words):
            # Calculate relevance score based on word proximity
            score = sum(text.count(word) for word in query_words)
            
            results.append({
                'text': segment['text'],
                'timestamp': timestamp,
                'score': score
            })
    
    # Sort by relevance score and limit to top 10
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:10]

@app.route('/search_transcript', methods=['POST'])
def search_transcript():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check if we have a transcript in the session
    transcript = session.get('transcript')
    if not transcript:
        return jsonify({'error': 'No transcript available. Please transcribe a video first.'}), 404
    
    # Get segments from session
    segments = session.get('timestamps', [])
    if not segments:
        # If no segments but we have transcript, create a single segment
        segments = [{
            'text': transcript,
            'start': 0
        }]
    
    results = search_transcript_segments(query, segments)
    return jsonify({'results': results})

def generate_summary(text, key_points=None):
    """Generate a concise summary from key points."""
    if not text:
        return ''

    # Extract key points if not provided
    if key_points is None:
        key_points = extract_key_points(text)

    # Format key points as bullet points
    if key_points:
        summary = '\n'.join(f'• {point}' for point in key_points)
    else:
        summary = text[:500] + '...' if len(text) > 500 else text
    
    return summary

if __name__ == "__main__":
    app.run(debug=True)
