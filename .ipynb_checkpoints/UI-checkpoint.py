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

#path to trained model
#MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'best_autoencoder.pth')
MODEL_PATH = 'C:/Users/deana/Downloads/RetinalDiseaseDetectionAutoencoder/src/best_autoencoder.pth'

#model arch
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

#loading trained modek
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model on {device}...')
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print('Model loaded successfully.')

#trainsforming with training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.147, 0.268, 0.425],
                         std= [0.141, 0.200, 0.282])
])

#current tested threshold
THRESHOLD = 0.13



#helper function to make tensors displayable
def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

#gauge to show how close score is to anamoly threshold
def make_gauge(score, is_anomaly):
    color = 'red' if is_anomaly else 'green'
    fig, ax = plt.subplots(figsize=(4.5, 1.2))
    ax.barh([''], [1.0], color='white', height=0.5)
    ax.barh([''], [min(score, 1.0)], color=color, height=0.5)
    ax.axvline(THRESHOLD, color='orange', linewidth=2, linestyle='--', label=f'Threshold ({THRESHOLD})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Reconstruction Error', fontsize=9)
    ax.set_title(f'Anomaly Score: {score:.4f}', fontsize=11, fontweight='bold', color=color)
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

#main function to detect anomalies
def analyze(image):
    if image is None:
        return None, None, "No image selected", ""

    pil_img = Image.fromarray(image).convert('RGB')
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(tensor)

    score = ((recon - tensor) ** 2).mean().item()
    is_anomaly = score > THRESHOLD
    print(f'Image: score={score:.6f} threshold={THRESHOLD} anomaly={is_anomaly}')

    recon_img = denorm(recon[0])
    gauge_img = make_gauge(score, is_anomaly)

    verdict = (
        f"Raw score: {score:.6f}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Decision: {'Has anomaly' if is_anomaly else 'Normal'}"
    )


    return (
        recon_img,
        gauge_img,
        verdict,
    )


# # test is title
# ** bold text
#http://localhost:7860
#gradio-app environment

with gr.Blocks(title='Retinal Anomaly Detector') as demo:

    gr.Markdown(""" # Retinal Disease Anomaly Dector """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_img = gr.Image(label='Upload Fundus Image', type='numpy', height=250)
            with gr.Row():
                run_btn = gr.Button('Analyze Image', variant='primary')

        with gr.Column(scale=1):
            gr.Markdown("### Reconstruction")
            recon_out = gr.Image(label='Autoencoder Reconstruction', type='numpy', height=220)
                
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            gauge_out = gr.Image(type='pil', height=140)
            verdict_out = gr.Textbox(label='Verdict', lines=3, interactive=False)


    run_btn.click(fn=analyze, inputs=[input_img], outputs=[recon_out, gauge_out, verdict_out])
    
if __name__ == '__main__':
    demo.launch(
    server_name='localhost',
    server_port=7860,
    share=False,
    show_error=True,
    theme=gr.themes.Soft()
)