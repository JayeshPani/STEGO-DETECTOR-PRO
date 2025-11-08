# ---- BEGIN HOTFIX: tolerate boolean JSON Schemas in gradio_client ----
try:
    import gradio_client.utils as _gcu
    _orig_get_type = _gcu.get_type
    _orig_json_to_py = _gcu._json_schema_to_python_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        return _orig_get_type(schema)
    def _safe_json_to_py(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_json_to_py(schema, defs)
    _gcu.get_type = _safe_get_type
    _gcu._json_schema_to_python_type = _safe_json_to_py
except Exception as _e:
    print("Gradio schema hotfix not applied:", _e)
# ---- END HOTFIX ----

import os, json, yaml, torch, cv2, numpy as np
import gradio as gr
import sys
sys.path.append("src")
from model import build_model
from transforms import val_tfms

CFG_PATH = "configs/train_full.yaml"
CKPT_PATH = "outputs/checkpoints/efnb0_best.pth"

# ---- Load model once ----
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
model = build_model(cfg["model"]).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
tfm = val_tfms()

def predict_image(np_image, threshold, logit_bias, temperature):
    import math  # local import to keep this function self-contained

    if np_image is None:
        return "No image provided.", "", ""

    # Ensure uint8 RGB
    if np_image.dtype != np.uint8:
        np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)
    if np_image.ndim == 2:
        rgb = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    elif np_image.shape[2] == 4:
        rgb = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
    else:
        rgb = np_image

    # Transforms -> tensor (CHW, float32)
    aug = tfm(image=rgb)
    img = aug["image"]
    if isinstance(img, np.ndarray):
        x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    elif isinstance(img, torch.Tensor):
        x = img.unsqueeze(0).float()
    else:
        return "Bad transform output.", "", ""

    x = x.to(device)

    # Forward
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        raw_logit = float(logits.view(-1)[0].item())

    # --- Calibration (bias + temperature) ---
    T = max(1e-6, float(temperature))
    b = float(logit_bias)
    calibrated_logit = (raw_logit - b) / T
    p = float(torch.sigmoid(torch.tensor(calibrated_logit)).item())

    # --- Decision ---
    thr = float(threshold)
    pred = int(p >= thr)
    label = "STEGANOGRAPHY" if pred == 1 else "CLEAN"

    # --- Symmetric, threshold-relative confidence ---
    def _logit(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return math.log(x / (1.0 - x))

    thr_logit = _logit(thr)
    margin = abs(calibrated_logit - thr_logit)  # distance in logit space from decision boundary
    k = 0.75  # controls how quickly confidence saturates with distance
    confidence = 1.0 - math.exp(-k * margin)   # 0 at boundary → 1 far away

    # Color coding for results
    color = "#FF6B6B" if pred == 1 else "#51CF66"

    details = {
        "raw_logit": raw_logit,
        "calibrated_logit": calibrated_logit,
        "stego_probability": p,
        "threshold": thr,
        "logit_bias": b,
        "temperature": T,
        "device": device,
        "confidence": confidence,
    }

    # Confidence bar HTML
    confidence_width = confidence * 100.0
    confidence_html = f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #94A3B8; font-size: 0.9rem;">Low Confidence</span>
            <span style="color: #94A3B8; font-size: 0.9rem;">High Confidence</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_width:.2f}%; background: linear-gradient(90deg, #FF6B6B, #51CF66);"></div>
        </div>
        <div style="text-align: center; margin-top: 10px;">
            <span style="color: {color}; font-weight: bold; font-size: 1.2rem; text-shadow: 0 0 10px {color}50;">
                Confidence: {confidence:.1%}
            </span>
        </div>
    </div>
    """

    return label, json.dumps(details, indent=2), confidence_html

# ---- Enhanced Futuristic UI with Randomized Bit Streams and Dark Gradient Background ----
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple"
    ),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
    
    body { 
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 25%, #0f0c29 50%, #2d2b55 75%, #24243e 100%) !important;
        font-family: 'Exo 2', sans-serif !important;
        overflow-x: hidden;
        position: relative;
        min-height: 100vh;
    }
    
    /* Enhanced dark gradient overlay */
    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(28, 227, 227, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(19, 125, 254, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(114, 9, 183, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -2;
    }
    
    /* Pure CSS Bit Stream Background - Larger Size with Randomized Layout */
    .bit-stream-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.25;
        overflow: hidden;
    }
    
    .bit-column {
        position: absolute;
        top: -100px;
        width: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }
    
    .bit {
        color: #1CE3E3;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 18px;
        opacity: 0;
        text-shadow: 0 0 15px currentColor;
        animation: bitFall 4s linear infinite;
    }
    
    .bit:nth-child(2n) { animation-delay: 0.5s; color: #137DFE; }
    .bit:nth-child(3n) { animation-delay: 1s; color: #3924B9; }
    .bit:nth-child(4n) { animation-delay: 1.5s; color: #66E3CA; }
    .bit:nth-child(5n) { animation-delay: 2s; color: #FF6B6B; }
    .bit:nth-child(6n) { animation-delay: 2.5s; color: #51CF66; }
    
    @keyframes bitFall {
        0% {
            transform: translateY(-100px) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 0.9;
        }
        90% {
            opacity: 0.7;
        }
        100% {
            transform: translateY(100vh) rotate(180deg);
            opacity: 0;
        }
    }
    
    /* Randomized column positions and animation delays */
    .bit-column:nth-child(1) { left: 5%; animation-delay: 0s; }
    .bit-column:nth-child(2) { left: 15%; animation-delay: 1.2s; }
    .bit-column:nth-child(3) { left: 22%; animation-delay: 0.3s; }
    .bit-column:nth-child(4) { left: 35%; animation-delay: 2.1s; }
    .bit-column:nth-child(5) { left: 48%; animation-delay: 0.8s; }
    .bit-column:nth-child(6) { left: 55%; animation-delay: 1.9s; }
    .bit-column:nth-child(7) { left: 68%; animation-delay: 0.5s; }
    .bit-column:nth-child(8) { left: 72%; animation-delay: 2.4s; }
    .bit-column:nth-child(9) { left: 85%; animation-delay: 1.1s; }
    .bit-column:nth-child(10) { left: 92%; animation-delay: 0.7s; }
    .bit-column:nth-child(11) { left: 28%; animation-delay: 1.8s; }
    .bit-column:nth-child(12) { left: 42%; animation-delay: 2.7s; }
    .bit-column:nth-child(13) { left: 65%; animation-delay: 0.9s; }
    .bit-column:nth-child(14) { left: 78%; animation-delay: 1.5s; }
    .bit-column:nth-child(15) { left: 8%; animation-delay: 2.3s; }
    
    /* Individual column animation variations */
    .bit-column:nth-child(1) .bit { animation-duration: 3.8s; }
    .bit-column:nth-child(2) .bit { animation-duration: 4.2s; }
    .bit-column:nth-child(3) .bit { animation-duration: 3.5s; }
    .bit-column:nth-child(4) .bit { animation-duration: 4.5s; }
    .bit-column:nth-child(5) .bit { animation-duration: 3.9s; }
    .bit-column:nth-child(6) .bit { animation-duration: 4.1s; }
    .bit-column:nth-child(7) .bit { animation-duration: 3.6s; }
    .bit-column:nth-child(8) .bit { animation-duration: 4.3s; }
    .bit-column:nth-child(9) .bit { animation-duration: 3.7s; }
    .bit-column:nth-child(10) .bit { animation-duration: 4.4s; }
    .bit-column:nth-child(11) .bit { animation-duration: 3.4s; }
    .bit-column:nth-child(12) .bit { animation-duration: 4.6s; }
    .bit-column:nth-child(13) .bit { animation-duration: 3.8s; }
    .bit-column:nth-child(14) .bit { animation-duration: 4.0s; }
    .bit-column:nth-child(15) .bit { animation-duration: 3.9s; }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .header-section {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, rgba(28, 58, 227, 0.15) 0%, rgba(114, 9, 183, 0.15) 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(28, 227, 227, 0.4);
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-family: 'Orbitron', monospace !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #1CE3E3 0%, #137DFE 50%, #3924B9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(28, 227, 227, 0.5);
        margin-bottom: 10px !important;
        letter-spacing: 2px;
    }
    
    .subtitle {
        color: #C4E0FF !important;
        font-size: 1.3rem !important;
        margin-bottom: 15px !important;
        font-weight: 300;
        text-shadow: 0 0 10px rgba(196, 224, 255, 0.3);
    }
    
    .tagline {
        color: #66E3CA !important;
        font-size: 1.1rem !important;
        font-weight: 400;
        text-shadow: 0 0 8px rgba(102, 227, 202, 0.3);
    }
    
    .gr-input, .gr-output, .gr-textbox, .gr-slider {
        border-radius: 16px !important;
        border: 1px solid #1CE3B1 !important;
        background: rgba(15, 20, 35, 0.95) !important;
        color: #FFFFFF !important;
        font-family: 'Exo 2', sans-serif !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .gr-input:focus, .gr-textbox:focus {
        border-color: #137DFE !important;
        box-shadow: 0 0 0 2px rgba(19, 125, 254, 0.5) !important;
        background: rgba(20, 25, 45, 0.95) !important;
    }
    
    .control-panel {
        background: linear-gradient(135deg, rgba(28, 58, 227, 0.15) 0%, rgba(114, 9, 183, 0.15) 100%) !important;
        border: 1px solid rgba(28, 227, 227, 0.3);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .panel-title {
        font-family: 'Orbitron', monospace !important;
        color: #1CE3E3 !important;
        font-size: 1.4rem !important;
        margin-bottom: 20px !important;
        text-align: center;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(28, 227, 227, 0.5);
    }
    
    .parameter-description {
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        margin-top: 5px !important;
        font-style: italic;
    }
    
    .gr-button {
        background: linear-gradient(135deg, #137DFE 0%, #3924B9 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        font-family: 'Orbitron', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(19, 125, 254, 0.4) !important;
        backdrop-filter: blur(10px);
    }
    
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(19, 125, 254, 0.6) !important;
        background: linear-gradient(135deg, #1CE3E3 0%, #137DFE 100%) !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(28, 58, 227, 0.2) 0%, rgba(114, 9, 183, 0.2) 100%) !important;
        border: 1px solid rgba(28, 227, 227, 0.4);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(15px);
        min-height: 200px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .confidence-bar {
        height: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin: 15px 0;
        border: 1px solid rgba(28, 227, 227, 0.3);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: all 0.5s ease;
        box-shadow: 0 0 20px rgba(28, 227, 227, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #62E3FA !important;
        opacity: 0.8;
        font-size: 0.95rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(28, 227, 227, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .upload-area {
        border: 2px dashed #1CE3B1 !important;
        border-radius: 20px !important;
        padding: 30px !important;
        text-align: center;
        background: rgba(15, 20, 35, 0.8) !important;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .upload-area:hover {
        border-color: #137DFE !important;
        background: rgba(20, 25, 45, 0.9) !important;
        box-shadow: 0 0 30px rgba(19, 125, 254, 0.3);
    }
    
    label, .gr-block-title {
        color: #1CE3E3 !important;
        font-weight: 600 !important;
        font-family: 'Exo 2', sans-serif !important;
        font-size: 1.1rem !important;
        text-shadow: 0 0 8px rgba(28, 227, 227, 0.3);
    }
    
    .gr-slider-value {
        color: #66E3CA !important;
        font-weight: 600 !important;
    }
    """
) as demo:
    
    # Enhanced Bit Stream Background with Randomized Layout
    gr.HTML("""
    <div class="bit-stream-background">
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">0</div>
        </div>
        <div class="bit-column">
            <div class="bit">0</div><div class="bit">0</div><div class="bit">1</div><div class="bit">1</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">0</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">1</div><div class="bit">0</div><div class="bit">1</div>
        </div>
        <div class="bit-column">
            <div class="bit">1</div><div class="bit">0</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">0</div><div class="bit">1</div><div class="bit">1</div><div class="bit">0</div>
            <div class="bit">1</div><div class="bit">0</div><div class="bit">0</div><div class="bit">1</div>
        </div>
    </div>
    """)
    
    # Header Section
    with gr.Column(elem_classes="header-section"):
        gr.HTML("""
        <div>
            <h1 class="main-title">🔮 STEGO DETECTOR PRO</h1>
            <p class="subtitle">Advanced Steganography Detection with Neural Networks</p>
            <p class="tagline">Uncover hidden data with precision • AI-powered analysis • Real-time results</p>
        </div>
        """)
    
    with gr.Row(equal_height=False):
        # Left Column - Image Upload
        with gr.Column(scale=6):
            with gr.Group():
                gr.Markdown("### 📷 Image Upload", elem_classes="panel-title")
                image_input = gr.Image(
                    type="numpy", 
                    label="Drag & Drop Image Here",
                    elem_classes="upload-area",
                    height=400
                )
        
        # Right Column - Controls
        with gr.Column(scale=4):
            with gr.Group(elem_classes="control-panel"):
                gr.Markdown("### ⚙️ Detection Parameters", elem_classes="panel-title")
                
                threshold = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.85, 
                    step=0.01, 
                    label="Decision Threshold",
                    info="Higher values = more conservative detection"
                )
                
                logit_bias = gr.Slider(
                    minimum=-3.0, 
                    maximum=3.0, 
                    value=1.0, 
                    step=0.05, 
                    label="Logit Bias",
                    info="Adjust model output bias (subtract from raw logits)"
                )
                
                temperature = gr.Slider(
                    minimum=0.5, 
                    maximum=3.0, 
                    value=1.0, 
                    step=0.05, 
                    label="Temperature",
                    info="Higher values = softer probability distribution"
                )
    
    # Analyze Button
    with gr.Row():
        with gr.Column():
            analyze_btn = gr.Button(
                "🚀 Analyze Image", 
                size="lg",
                variant="primary"
            )
    
    # Results Section
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="result-card"):
                gr.Markdown("### 📊 Analysis Results", elem_classes="panel-title")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prediction = gr.Textbox(
                            label="Prediction",
                            show_label=True,
                            elem_classes="gr-output"
                        )
                
                # Confidence bar visualization - now without the slider
                confidence_html = gr.HTML("""
                <div style="margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #94A3B8; font-size: 0.9rem;">Low Confidence</span>
                        <span style="color: #94A3B8; font-size: 0.9rem;">High Confidence</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 50%; background: linear-gradient(90deg, #FF6B6B, #51CF66);"></div>
                    </div>
                    <div style="text-align: center; margin-top: 10px;">
                        <span style="color: #94A3B8; font-weight: bold; font-size: 1.1rem;">
                            Upload an image to see confidence level
                        </span>
                    </div>
                </div>
                """)
        
        with gr.Column():
            with gr.Group(elem_classes="result-card"):
                gr.Markdown("### 🔍 Technical Details", elem_classes="panel-title")
                details = gr.Textbox(
                    label="Analysis Details",
                    lines=8,
                    max_lines=12,
                    show_label=False,
                    elem_classes="gr-output"
                )
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>🚀 Powered by <b>EfficientNet-B0</b> • 🛡️ Enterprise-Grade Steganography Detection</p>
        <p>⚡ Real-time Analysis • 🎯 Precision Calibration • 🔮 Future-Ready Technology</p>
    </div>
    """)
    
    # Interactive function call - removed confidence slider output
    analyze_btn.click(
        fn=predict_image,
        inputs=[image_input, threshold, logit_bias, temperature],
        outputs=[prediction, details, confidence_html]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=True, 
        show_api=False
    )