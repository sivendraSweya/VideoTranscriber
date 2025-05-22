from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, PageTemplate, Frame, NextPageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepTogether, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def analyze_transcript(transcript):
    """Analyze the transcript to extract key points, action items, and topics."""
    # Tokenize into sentences
    sentences = sent_tokenize(transcript)
    
    # Extract action items (sentences with action verbs)
    action_verbs = ['need', 'must', 'should', 'will', 'going to', 'plan to', 
                   'decide', 'agree', 'propose', 'suggest', 'recommend', 'action',
                   'task', 'todo', 'assign', 'deadline', 'due', 'follow up']
    
    action_items = []
    for sentence in sentences:
        if any(verb in sentence.lower() for verb in action_verbs):
            # Clean up the sentence
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            if len(clean_sentence.split()) > 3:  # Only include meaningful sentences
                action_items.append(clean_sentence)
    
    # Extract key discussion points (sentences with important nouns)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(transcript.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)  # Get most common words
    
    # Find sentences containing key terms
    key_points = []
    for sentence in sentences:
        # Check if sentence contains any of the common important words
        sentence_words = set(word_tokenize(sentence.lower()))
        if any(word in sentence_words for word, _ in common_words[:10]):
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            if len(clean_sentence.split()) > 5:  # Only include meaningful sentences
                key_points.append(clean_sentence)
    
    # Remove duplicates while preserving order
    key_points = list(dict.fromkeys(key_points))
    
    # Extract topics (most common nouns)
    pos_tags = nltk.pos_tag(word_tokenize(transcript))
    nouns = [word for word, pos in pos_tags if pos.startswith('NN') and word.lower() not in stop_words]
    topics = [word for word, _ in Counter(nouns).most_common(5)]
    
    return {
        'action_items': action_items[:10],  # Limit to top 10
        'key_points': key_points[:15],     # Limit to top 15
        'topics': topics                   # Top 5 topics
    }

def create_pdf(transcript_text, output_filename, title="Meeting Transcript", analyze_content=True):
    """
    Create a professional PDF from transcript text
    
    Args:
        transcript_text (str): The transcript text to convert to PDF
        output_filename (str): Full path where to save the PDF file
        title (str): Title for the document
    """
    output_path = output_filename
    
    # Create document with margins
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    # Custom style names to avoid conflicts
    custom_styles = {}
    
    # Title style
    if 'CustomTitle' not in styles:
        custom_styles['CustomTitle'] = ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
    
    # Subtitle style
    if 'CustomSubtitle' not in styles:
        custom_styles['CustomSubtitle'] = ParagraphStyle(
            name='CustomSubtitle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=40,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
    
    # Header style
    if 'CustomHeader' not in styles:
        custom_styles['CustomHeader'] = ParagraphStyle(
            name='CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        )
    
    # Subheader style
    if 'CustomSubheader' not in styles:
        custom_styles['CustomSubheader'] = ParagraphStyle(
            name='CustomSubheader',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        )
    
    # Highlight style
    if 'CustomHighlight' not in styles:
        custom_styles['CustomHighlight'] = ParagraphStyle(
            name='CustomHighlight',
            parent=styles['Normal'],
            backColor=colors.HexColor('#f8f9fa'),
            borderWidth=1,
            borderColor=colors.HexColor('#dee2e6'),
            borderPadding=10,
            spaceBefore=10,
            spaceAfter=10
        )
    
    # Bullet style
    if 'CustomBullet' not in styles:
        custom_styles['CustomBullet'] = ParagraphStyle(
            name='CustomBullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-10,
            spaceBefore=5,
            bulletIndent=0,
            bulletFontSize=10
        )
    
    # Add all custom styles to the stylesheet
    for name, style in custom_styles.items():
        if name not in styles:
            styles.add(style)
    
    # Create story (content elements)
    story = []
    
    # Add title page
    def add_title_page():
        nonlocal story
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph(title, styles['CustomTitle']))
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph("Meeting Transcript", styles['CustomSubtitle']))
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles['CustomSubtitle']))
        story.append(PageBreak())
    
    # Add table of contents
    def add_toc():
        nonlocal story
        story.append(Paragraph("Table of Contents", styles['CustomHeader']))
        story.append(Spacer(1, 20))
        # This would be populated with actual content later
        story.append(Paragraph("1. Meeting Overview ................. 3", styles['Normal']))
        story.append(Paragraph("2. Key Discussion Points .......... 4", styles['Normal']))
        story.append(Paragraph("3. Action Items ................... 5", styles['Normal']))
        story.append(PageBreak())
    
    # Add content
    def add_content(analysis=None):
        nonlocal story, transcript_text
        
        # Default empty analysis if not provided
        if analysis is None:
            analysis = {'action_items': [], 'key_points': [], 'topics': []}
        
        # 1. Executive Summary
        story.append(Paragraph("1. Executive Summary", styles['CustomHeader']))
        story.append(Spacer(1, 12))
        
        # Add meeting metadata
        highlight_text = f"""
        <b>Document Title:</b> {title}<br/>
        <b>Date Generated:</b> {datetime.now().strftime("%B %d, %Y %H:%M")}<br/>
        <b>Transcript Length:</b> {len(transcript_text.split())} words, {len(transcript_text.split('.'))} sentences
        """
        story.append(Paragraph(highlight_text, styles['CustomHighlight']))
        
        # Add key topics if available
        if analysis.get('topics'):
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Key Topics:</b>", styles['Normal']))
            for topic in analysis['topics']:
                story.append(Paragraph(f"• {topic.capitalize()}", styles['CustomBullet']))
        
        # 2. Key Discussion Points
        story.append(PageBreak())
        story.append(Paragraph("2. Key Discussion Points", styles['CustomHeader']))
        
        if analysis.get('key_points'):
            for i, point in enumerate(analysis['key_points'], 1):
                story.append(Paragraph(f"{i}. {point}", styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            # Fallback to simple paragraph split if no analysis
            paragraphs = [p.strip() for p in transcript_text.split('\n') if p.strip()]
            for i, para in enumerate(paragraphs[:15], 1):  # Limit to first 15 paragraphs
                story.append(Paragraph(f"{i}. {para}", styles['Normal']))
                story.append(Spacer(1, 8))
        
        # 3. Action Items
        if analysis.get('action_items'):
            story.append(PageBreak())
            story.append(Paragraph("3. Action Items", styles['CustomHeader']))
            
            # Create action items table
            action_items = []
            for i, item in enumerate(analysis['action_items'], 1):
                # Simple extraction of potential assignee (first proper noun in the sentence)
                words = word_tokenize(item)
                pos_tags = nltk.pos_tag(words)
                assignee = next((word for word, pos in pos_tags 
                              if pos in ['NNP', 'NNPS'] and word.istitle()), "Unassigned")
                
                # Add to action items
                action_items.append((
                    f"AI-{i:03d}",  # Action Item ID
                    item[:100] + (item[100:] and '...'),  # Truncate long items
                    assignee,
                    "",  # Due date (would need date extraction)
                    "Pending"  # Status
                ))
            
            if action_items:
                # Create table
                data = [
                    ["ID", "Action Item", "Assigned To", "Due Date", "Status"],
                    *action_items
                ]
                
                # Style the table
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center align ID column
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ]
                
                # Calculate column widths (ID: 10%, Action: 50%, Assigned: 15%, Due: 15%, Status: 10%)
                col_widths = [
                    doc.width * 0.1,  # ID
                    doc.width * 0.5,   # Action Item
                    doc.width * 0.15,  # Assigned To
                    doc.width * 0.15,  # Due Date
                    doc.width * 0.1    # Status
                ]
                
                # Create and style the table
                table = Table(data, colWidths=col_widths, repeatRows=1)
                table.setStyle(TableStyle(table_style))
                story.append(table)
            else:
                story.append(Paragraph("No specific action items were identified in the transcript.", styles['Italic']))
        
        # 4. Full Transcript (if there's space)
        if len(transcript_text) < 5000:  # Only include full transcript if it's not too long
            story.append(PageBreak())
            story.append(Paragraph("4. Full Transcript", styles['CustomHeader']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(transcript_text, styles['Normal']))
    
    # Add footer with page numbers
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(doc.width + doc.rightMargin/2, 0.4*inch, text)
        canvas.drawString(doc.leftMargin, 0.4*inch, "Confidential")
        canvas.restoreState()
    
    # Analyze the transcript if enabled
    analysis = {}
    if analyze_content and transcript_text.strip():
        try:
            analysis = analyze_transcript(transcript_text)
        except Exception as e:
            print(f"Error analyzing transcript: {str(e)}")
    
    # Build the document
    add_title_page()
    add_toc()
    add_content(analysis)
    
    # Add header and footer
    def add_header_footer(canvas, doc):
        canvas.saveState()
        
        # Add header
        header = f"{title} - {datetime.now().strftime('%B %d, %Y')}"
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(doc.width + doc.rightMargin/2, doc.height + doc.topMargin - 15, header)
        
        # Add footer with page number
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(doc.width + doc.rightMargin/2, 0.4*inch, text)
        canvas.drawString(doc.leftMargin, 0.4*inch, "Confidential")
        
        # Add a line
        canvas.setStrokeColor(colors.HexColor('#e0e0e0'))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 5, 
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 5)
        
        canvas.restoreState()
    
    # Build the PDF with header and footer
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    
    print(f"PDF created successfully at: {os.path.abspath(output_path)}")
    return output_path

# Example usage
if __name__ == "__main__":
    # Sample transcript text
    sample_transcript = """
    Welcome everyone to our quarterly planning meeting. Let's start with a quick round of introductions.
    
    I'm John, the project manager for the new client portal. We've made great progress on the backend services.
    
    Hi, I'm Sarah from the design team. We've been working on the UI components and have some mockups to share.
    
    The main challenges we're facing are related to the API response times when handling large datasets.
    
    We've identified three key areas for improvement: database optimization, caching strategies, and frontend pagination.
    
    The timeline for implementing these changes is approximately six weeks, with bi-weekly check-ins.
    
    Let's break this down into smaller tasks and assign owners for each component.
    """
    
    create_pdf(sample_transcript, "sample_transcript.pdf", "Q4 Planning Meeting")
    print("Sample PDF created. You can now use this script with your own transcript text.")
