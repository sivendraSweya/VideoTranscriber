from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_from_directory, flash
import re
from werkzeug.utils import secure_filename
import os
import whisper
import requests
import json
import socket
import tempfile
from datetime import datetime
from transcript_to_pdf import create_pdf


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
# Use a secure secret key in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# Configure session
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


# Ensure upload folder exists
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Using Flask's built-in session

model = whisper.load_model("base")


@app.route("/")
def index():
    return render_template(
        "index.html",
        transcript=session.get("transcript", ""),
        action_points=session.get("action_points", "")
    )


def has_audio_stream(filepath):
    """Check if the video file contains an audio stream using ffprobe."""
    import subprocess
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'audio' in result.stdout.lower()
    except Exception as e:
        print(f"Error checking audio stream: {e}")
        return False


@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("\n=== Starting Transcription ===")
    print(f"Request form: {request.form}")
    print(f"Request files: {request.files}")
    
    if 'file' not in request.files:
        error_msg = 'No file part in request'
        print(f"Error: {error_msg}")
        flash(error_msg, 'error')
        return redirect(request.referrer or url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file:
        return jsonify({'error': 'No file provided'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        
        # Check if file has audio
        if not has_audio_stream(filepath):
            os.remove(filepath)  # Clean up the file
            return jsonify({'error': 'The uploaded file does not contain an audio stream'}), 400

        # Transcribe the audio
        result = model.transcribe(filepath)
        transcript = result["text"]
        
        # Instead of storing in session, save to a file and store the filename
        transcript_filename = f"transcript_{int(datetime.now().timestamp())}.txt"
        transcript_path = os.path.join('transcripts', transcript_filename)
        
        # Ensure transcripts directory exists
        os.makedirs('transcripts', exist_ok=True)
        
        # Save transcript to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Store only the filename in session
        session['transcript_file'] = transcript_filename
        session['filename'] = os.path.splitext(filename)[0]  # Save filename without extension
        session['transcript_length'] = len(transcript)
        session.modified = True  # Ensure session is saved
        
        # Debug output
        print("\n=== SESSION DATA SAVED ===")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Session keys: {list(session.keys())}")
        print("=========================\n")
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
            print("Temporary file cleaned up")
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {str(e)}")
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'status': 'success',
                'redirect': url_for('chat')
            })
        return redirect(url_for('chat'))
            
    except Exception as e:
        import traceback
        error_msg = f'Error: {str(e)}'
        print(f"\n=== ERROR ===")
        print(error_msg)
        print("\nStack Trace:")
        print(traceback.format_exc())
        print("=============\n")
        
        # Clean up if file exists
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print("Cleaned up temporary file after error")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
        
        flash(f'Error processing file: {str(e)}', 'error')
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500
            
        return redirect(request.referrer or url_for('index'))


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    """Generate a PDF from the transcript file with content analysis."""
    print("\n=== Starting PDF Generation ===")
    
    if 'transcript_file' not in session or not session['transcript_file']:
        error_msg = "No transcript file found. Please upload a file first."
        print(f"Error: {error_msg}")
        flash(error_msg, 'error')
        return redirect(url_for('index'))
    
    try:
        # Load transcript from file
        transcript_path = os.path.join('transcripts', session['transcript_file'])
        if not os.path.exists(transcript_path):
            error_msg = "Transcript file not found. Please upload the file again."
            print(f"Error: {error_msg}")
            flash(error_msg, 'error')
            return redirect(url_for('index'))
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Get custom title or use default
        custom_title = request.form.get('title', 'Meeting Transcript')
        safe_title = re.sub(r'[^\w\s-]', '', custom_title).strip().replace(' ', '_')
        
        # Create a safe filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{safe_title}_{timestamp}.pdf"
        output_path = os.path.abspath(os.path.join('output', output_filename))
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        print(f"Generating PDF with title: {custom_title}")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Output path: {output_path}")
        
        # Generate the PDF
        from transcript_to_pdf import create_pdf
        
        try:
            pdf_path = create_pdf(
                transcript_text=transcript,
                output_filename=output_path,
                title=custom_title,
                analyze_content=True
            )
            
            print(f"PDF generated at: {pdf_path}")
            
            # Verify the file was created
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Failed to generate PDF file at {pdf_path}")
            
            # Get the relative path for send_from_directory
            output_dir = os.path.abspath('output')
            if not output_path.startswith(output_dir):
                raise ValueError(f"Invalid output path: {output_path}")
                
            rel_path = os.path.relpath(pdf_path, output_dir)
            
            # Send the file for download
            return send_from_directory(
                directory=output_dir,
                path=os.path.basename(rel_path),
                as_attachment=True,
                download_name=f"{safe_title}.pdf"
            )
            
        except Exception as e:
            error_msg = f"Error in PDF generation: {str(e)}"
            print(f"\n=== PDF GENERATION ERROR ===\n{error_msg}\n")
            import traceback
            traceback.print_exc()
            print("==============================\n")
            raise
        
    except Exception as e:
        error_msg = f"Error generating PDF: {str(e)}"
        print(f"\n=== ERROR ===\n{error_msg}\n")
        import traceback
        traceback.print_exc()
        print("=============\n")
                
        flash('Failed to generate PDF. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/chat')
def chat():
    # Debug: Print all session keys
    print("\n=== Chat Route ===")
    print("Session keys:", list(session.keys()))
    
    # Check if transcript file exists in session
    if 'transcript_file' not in session or not session['transcript_file']:
        print("No transcript file found in session. Redirecting to home.")
        flash('No transcript found. Please upload a file first.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Load transcript from file
        transcript_path = os.path.join('transcripts', session['transcript_file'])
        if not os.path.exists(transcript_path):
            flash('Transcript file not found. Please upload the file again.', 'error')
            return redirect(url_for('index'))
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
            
        # Get action points if they exist
        action_points = session.get('action_points', '')
        
        # Get suggestions
        suggestions = [
            "Summarize the key points",
            "What are the main topics discussed?",
            "Extract action items",
            "What decisions were made?"
        ]
        
        return render_template('chat_new.html', 
                             transcript=transcript, 
                             action_points=action_points,
                             suggestions=suggestions,
                             filename=session.get('filename', 'Untitled'),
                             transcript_length=session.get('transcript_length', 0))
                             
    except Exception as e:
        print(f"Error loading transcript: {str(e)}")
        flash('Error loading transcript. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if we have a transcript file reference in session
    if 'transcript_file' not in session:
        return jsonify({'error': 'No transcript found. Please upload a file first.'}), 400

    try:
        # Load the transcript from file
        transcript_path = os.path.join('transcripts', session['transcript_file'])
        if not os.path.exists(transcript_path):
            return jsonify({'error': 'Transcript file not found. Please upload the file again.'}), 404

        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Here you would typically send the question and transcript to your LLM
        # For now, we'll just return a simple response
        response = (
            f"You asked: {question}\n\n"
            f"This is a placeholder response based on a transcript of "
            f"{len(transcript)} characters. In a real implementation, this would be "
            "the AI's response based on the transcript content."
        )

        return jsonify({
            'success': True,
            'response': response
        })

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON data'}), 400
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return jsonify({
            'error': f'Error processing question: {str(e)}'
        }), 500

# Initialize upload folder and run the app
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5000)
