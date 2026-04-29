import pandas as pd
import json
import whois
from urllib.parse import urlparse
from datetime import datetime
import requests
import gradio as gr
import ollama

import pickle

# ==========================================
# PART 1: SMS Spam Detection Model Setup
# ==========================================
def load_local_model():
    print("Loading local Naive Bayes model and vectorizer from .pkl files...")
    try:
        with open("spam_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("spam_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading local model: {e}")
        return None, None

model, vectorizer = load_local_model()

# ==========================================
# PART 2: Local Ollama LLM Configuration
# ==========================================
TEXT_MODEL   = 'phi3'
VISION_MODEL = 'moondream'

def ollama_chat(prompt, model=TEXT_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

def ollama_vision(prompt, image_path, model=VISION_MODEL):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_data]
        }]
    )
    return response['message']['content']

def trace_redirects(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        return response.url
    except:
        return url

# ==========================================
# PART 3: Gradio App Interface
# ==========================================

def smishx_app(sms_text, image_path):
    if not sms_text:
        return "Please enter an SMS text.", ""
    log = f"📡 INTERCEPTED SMS: '{sms_text}'\n" + "="*50 + "\n"
    if model and vectorizer:
        custom_message_vec = vectorizer.transform([sms_text])
        prediction = model.predict(custom_message_vec)
        local_result = "SPAM" if prediction[0] == 1 else "SAFE"
        log += f"   - Local Model Prediction: {local_result}\n\n"
    else:
        log += "   - Local Model Unavailable.\n\n"
    extract_prompt = (f'Extract Brand and URL from this SMS message. Return ONLY a valid JSON object with two keys: "brand" and "url". If a value is not present, use null. Do NOT include any explanation or markdown. SMS: "{sms_text}"')
    brand, url = None, None
    try:
        raw_data = ollama_chat(extract_prompt).strip()
        start = raw_data.find('{')
        end = raw_data.rfind('}') + 1
        data = json.loads(raw_data[start:end])
        brand = data.get("brand")
        url = data.get("url")
        log += f"   - Targeted Brand: {brand}\n   - Suspicious URL: {url}\n"
    except Exception as e:
        log += f"   - Extraction Error: {str(e)}\n"
    domain_age_days = "Unknown"
    final_url = url
    if url:
        final_url = trace_redirects(url)
        try:
            domain = urlparse(final_url).netloc
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list): creation_date = creation_date[0]
            if creation_date:
                domain_age_days = (datetime.now() - creation_date).days
                log += f"   - Domain: {domain}\n   - Domain Age: {domain_age_days} days\n"
        except Exception as e:
            log += f"   - WHOIS Lookup Failed: {e}\n"
    
    verdict_prompt = f"Analyze this SMS: {sms_text}. Brand: {brand}. URL: {final_url}. Age: {domain_age_days}. Provide PHISHING DETECTED or SAFE verdict."
    vision_analysis = ""
    if image_path:
        try:
            vision_analysis = ollama_vision("Analyze this screenshot for phishing features.", image_path)
            log += "   - Vision scan complete.\n"
        except Exception as e: log += f"   - Vision scan failed: {e}\n"
    try:
        technical_analysis = ollama_chat(verdict_prompt)
        final_verdict = ollama_chat(f"Create a user alert based on: {technical_analysis}. Vision context: {vision_analysis}")
    except Exception as e: return f"Error: {str(e)}", log
    return final_verdict, log

# --- GRADIO INTERFACE ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body, .gradio-container {
    background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #020617 80%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Glassmorphic Header */
.header-banner {
    background: rgba(30, 58, 138, 0.2);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(96, 165, 250, 0.2);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 30px rgba(59, 130, 246, 0.1);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.header-banner h1 {
    color: #ffffff;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    text-shadow: 0 0 15px rgba(255,255,255,0.3);
}
.header-banner p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* 3D Glass Panels for Columns */
.glass-panel {
    background: rgba(15, 23, 42, 0.5) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 15px 30px rgba(0,0,0,0.6) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* Input Fields styling */
.glass-panel textarea, .glass-panel input, .upload-container {
    background: rgba(2, 6, 23, 0.7) !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.8) !important;
}
.glass-panel textarea:focus, .glass-panel input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3), inset 0 2px 4px rgba(0,0,0,0.8) !important;
}

/* 3D Glowing Button */
button.primary {
    background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%) !important;
    border: 2px solid #93c5fd !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.5), inset 0 2px 5px rgba(255, 255, 255, 0.4), 0 5px 15px rgba(0,0,0,0.5) !important;
    border-radius: 9999px !important; /* Fully rounded pill */
    color: white !important;
    font-weight: bold !important;
    font-size: 1.2rem !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    margin-top: 1rem !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.8), inset 0 2px 5px rgba(255, 255, 255, 0.6), 0 5px 15px rgba(0,0,0,0.5) !important;
    transform: translateY(-1px) !important;
    background: linear-gradient(180deg, #60a5fa 0%, #2563eb 100%) !important;
}
button.primary:active {
    transform: translateY(2px) !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5), inset 0 3px 6px rgba(0,0,0,0.6) !important;
}

/* SIEM Logs style */
.log-box textarea {
    font-family: 'Courier New', monospace !important;
    background-color: #000000 !important;
    color: #10b981 !important;
    border: 1px solid #064e3b !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,1) !important;
}

/* Headings */
.prose h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem !important;
}

/* Tabs */
.tab-nav {
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
}
.tab-nav button {
    color: #94a3b8 !important;
    border: none !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #60a5fa !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    border-bottom: none !important;
    background: rgba(30, 58, 138, 0.3) !important;
    border-radius: 8px 8px 0 0 !important;
}
"""

theme = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px"
)

with gr.Blocks() as interface:
    gr.HTML('''
    <div class="header-banner">
        <h1>🛡️ Spam Shield Vision</h1>
        <p>Enterprise-Grade Multimodal Phishing & Smishing Detector</p>
    </div>
    ''')
    
    with gr.Row():
        with gr.Column(scale=4, elem_classes=["glass-panel"]):
            gr.Markdown("### 🔍 Threat Analysis Input")
            gr.Markdown("Provide the suspicious message and optional visual evidence for analysis.")
            
            sms_input = gr.Textbox(
                label="📱 Suspicious SMS Text", 
                placeholder="e.g., URGENT: Your package from FedEx is delayed.\nClick here to update delivery fee:\nhttp://fedex-update-tracker-xyz.com", 
                lines=4
            )
            image_input = gr.Image(
                type="filepath", 
                label="📸 Screenshot of Landing Page (Optional)"
            )
            
            submit_btn = gr.Button("🛡️ Analyze Threat", variant="primary", size="lg")
            
        with gr.Column(scale=5, elem_classes=["glass-panel"]):
            gr.Markdown("### 📊 Threat Intelligence Report")
            gr.Markdown("Real-time analysis from AI and local heuristic engines.")
            
            with gr.Tabs():
                with gr.TabItem("🛡️ Final Verdict"):
                    verdict_output = gr.Textbox(
                        label="Executive Summary", 
                        interactive=False, 
                        lines=14
                    )
                with gr.TabItem("⚙️ SIEM Logs"):
                    log_output = gr.Textbox(
                        label="Technical Execution Logs", 
                        interactive=False, 
                        lines=16,
                        elem_classes=["log-box"]
                    )

    submit_btn.click(
        fn=smishx_app, 
        inputs=[sms_input, image_input], 
        outputs=[verdict_output, log_output],
        api_name="analyze"
    )

if __name__ == "__main__":
    interface.launch(share=True, theme=theme, css=custom_css)