# app.py — Retinal Anomaly Detector
# Run with: python app.py
# Requires: pip install gradio torch torchvision pillow matplotlib numpy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
import gradio as gr
import datetime
import io
import os

#http://localhost:7860
#C:\Users\deana\Downloads\RetinalDiseaseDetectionAutoencoder>python app.py

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__),
             'src', 'best_autoencoder.pth')

# ── model definition (must match training exactly) ────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32,  64,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64,  32,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32,  3,   kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

# ── load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model on {device}...')
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print('Model loaded successfully.')

# ── transform (must match training exactly) ───────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.147, 0.268, 0.425],
                         std= [0.141, 0.200, 0.282])
])

THRESHOLD = 0.13
history   = []

# ── helpers ───────────────────────────────────────────────────────────────────
def denorm(tensor):
    """Denormalize tensor to displayable uint8 numpy array."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

def make_heatmap(orig_t, recon_t):
    """Generate blended reconstruction error heatmap."""
    err   = ((orig_t - recon_t) ** 2).mean(dim=0).cpu().numpy()
    err   = (err - err.min()) / (err.max() - err.min() + 1e-8)
    heat = (matplotlib.colormaps['hot'](err)[:, :, :3] * 255).astype(np.uint8)
    orig  = denorm(orig_t)
    blend = (0.5 * orig + 0.5 * heat).astype(np.uint8)
    return orig, blend

def make_gauge(score, is_anomaly):
    """Generate anomaly score gauge as PIL image."""
    color = '#e74c3c' if is_anomaly else '#2ecc71'
    fig, ax = plt.subplots(figsize=(4.5, 1.2))
    ax.barh([''], [1.0],              color='#eeeeee', height=0.5)
    ax.barh([''], [min(score, 1.0)],  color=color,     height=0.5)
    ax.axvline(THRESHOLD, color='#e67e22', linewidth=2,
               linestyle='--', label=f'Threshold ({THRESHOLD})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Reconstruction Error', fontsize=9)
    ax.set_title(f'Anomaly Score: {score:.4f}', fontsize=11,
                 fontweight='bold', color=color)
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(left=False, labelleft=False)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── inference ─────────────────────────────────────────────────────────────────
def analyze(image, eye_side, patient_id):
    if image is None:
        return None, None, None, "⚠️ Please upload an image.", "", get_history()

    pil_img   = Image.fromarray(image).convert('RGB')
    tensor    = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(tensor)

    score      = ((recon - tensor) ** 2).mean().item()
    is_anomaly = score > THRESHOLD
    print(f'Image: score={score:.6f}  threshold={THRESHOLD}  anomaly={is_anomaly}')

    orig_img, blended = make_heatmap(tensor[0], recon[0])
    recon_img         = denorm(recon[0])
    gauge_img         = make_gauge(score, is_anomaly)

    verdict = (
        f"⚠️  ANOMALY DETECTED\n"
        f"Score {score:.4f} exceeds threshold {THRESHOLD}"
        if is_anomaly else
        f"✅  NORMAL\n"
        f"Score {score:.4f} is within normal range"
    )

    details = (
        f"Raw score   : {score:.6f}\n"
        f"Threshold   : {THRESHOLD}\n"
        f"Eye side    : {eye_side}\n"
        f"Device      : {str(device).upper()}\n"
        f"Decision    : {'ANOMALY' if is_anomaly else 'NORMAL'}"
    )

    pid   = patient_id.strip() or '—'
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    history.insert(0, [stamp, pid, eye_side,
                        f'{score:.4f}',
                        '⚠️ Anomaly' if is_anomaly else '✅ Normal'])

    return (
        recon_img,
        blended,
        gauge_img,
        verdict,
        details,
        get_history()
    )

def get_history():
    return history[:20] if history else [['—', '—', '—', '—', '—']]

def clear_history():
    history.clear()
    return get_history()

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title='Retinal Anomaly Detector') as demo:

    gr.Markdown("""
    # 🔬 Retinal Anomaly Detector
    **Label-Free Ocular Disease Detection via Convolutional Autoencoder**

    Upload a fundus image to receive an anomaly score. The model was trained exclusively
    on healthy retinas — images it cannot reconstruct well are flagged as anomalous.
    > ⚕️ *For research purposes only. Not a clinical diagnostic tool.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            input_img  = gr.Image(label='Upload Fundus Image',
                                  type='numpy', height=250)
            eye_side   = gr.Radio(['Left', 'Right', 'Unknown'],
                                  label='Eye Side', value='Unknown')
            patient_id = gr.Textbox(label='Patient ID (optional)',
                                    placeholder='e.g. P-001')
            with gr.Row():
                run_btn   = gr.Button('🔍  Analyze', variant='primary')
                clear_btn = gr.Button('🗑  Clear History', variant='secondary')

        with gr.Column(scale=1):
            gr.Markdown("### 🖼 Image Preview")
            recon_out   = gr.Image(label='Autoencoder Reconstruction',
                                   type='numpy', height=220)
            heatmap_out = gr.Image(label='Reconstruction Error Heatmap',
                                   type='numpy', height=220)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            gauge_out   = gr.Image(label='Anomaly Score Gauge',
                                   type='pil', height=140)
            verdict_out = gr.Textbox(label='Verdict', lines=2,
                                     interactive=False)
            detail_out  = gr.Textbox(label='Details', lines=5,
                                     interactive=False)

    gr.Markdown("---\n### 📋 Session History")
    history_tbl = gr.Dataframe(
        headers=['Time', 'Patient ID', 'Eye', 'Score', 'Status'],
        value=get_history(),
        interactive=False,
        row_count=8
    )

    gr.Markdown("""
    ---
    **Interpreting results:**
    - **Score < 0.530** → Reconstruction error within normal range → ✅ Normal
    - **Score ≥ 0.530** → High reconstruction error → ⚠️ Refer for expert review
    - The heatmap highlights regions where the model struggled to reconstruct
    - Mean AUROC: 0.570 across 7 disease classes | Myopia AUROC: 0.768
    """)

    run_btn.click(
        fn=analyze,
        inputs=[input_img, eye_side, patient_id],
        outputs=[recon_out, heatmap_out, gauge_out,
                 verdict_out, detail_out, history_tbl]
    )
    clear_btn.click(fn=clear_history, outputs=[history_tbl])

# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    demo.launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=False,
    show_error=True,
    theme=gr.themes.Soft()
)