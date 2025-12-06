from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from typing import Dict, List, Tuple
import io
import base64
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMSIZE = 512 if DEVICE.type == 'cuda' else 256
if DEVICE.type == 'cuda':
    print(f'CUDA device (GPU): {torch.cuda.get_device_name(0)}')

# Normalization constants
NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)

# Layer mappings
STYLE_LAYER_MAP = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1',
}
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYER = 'conv4_2'

# Load VGG19 model
def load_vgg19_features() -> torch.nn.Sequential:
    weights = models.VGG19_Weights.DEFAULT
    vgg = models.vgg19(weights=weights).features.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg
VGG_FEATURES = load_vgg19_features()

# Image processing functions
def load_image_from_bytes(image_bytes: bytes, max_size: int = IMSIZE) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(DEVICE)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clone().clamp(0, 1)
    image = image.squeeze(0)
    to_pil = transforms.ToPILImage()
    return to_pil(image)

def normalize_for_vgg(tensor: torch.Tensor) -> torch.Tensor:
    mean = NORMALIZE_MEAN.view(1, -1, 1, 1)
    std = NORMALIZE_STD.view(1, -1, 1, 1)
    return (tensor - mean) / std

# Feature extraction and Gram matrix
def get_features(image: torch.Tensor, model: torch.nn.Sequential, layer_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    x = normalize_for_vgg(image)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_map:
            features[layer_map[name]] = x
    return features

def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = tensor.size()
    features = tensor.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)

# Loss functions
def calculate_content_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(generated, target)

def calculate_style_loss(generated: torch.Tensor, target_gram: torch.Tensor) -> torch.Tensor:
    generated_gram = gram_matrix(generated)
    return F.mse_loss(generated_gram, target_gram)

# Style transfer optimization loop
def run_style_transfer(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    num_steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    learning_rate: float = 0.02,
) -> torch.Tensor:
    style_features = get_features(style_image, VGG_FEATURES, STYLE_LAYER_MAP)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}
    content_features = get_features(content_image, VGG_FEATURES, STYLE_LAYER_MAP)
    target_content = content_features[CONTENT_LAYER]

    generated = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=learning_rate)
    log_every = max(num_steps // 10, 1)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        generated_features = get_features(generated, VGG_FEATURES, STYLE_LAYER_MAP)
        content_loss = calculate_content_loss(generated_features[CONTENT_LAYER], target_content)
        style_loss = 0.0
        for layer in STYLE_LAYERS:
            style_loss = style_loss + calculate_style_loss(generated_features[layer], style_grams[layer])
        style_loss = style_loss / len(STYLE_LAYERS)
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated.clamp_(0, 1)

        if step % log_every == 0 or step == 1:
            print(
                f'Step {step:4d}/{num_steps} | '
                f'Content: {content_loss.item():.2f} '
                f'Style: {style_loss.item()} '
                f'Total: {total_loss.item():.2f}'
            )

    return generated.detach()

# API endpoints
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "image_size": IMSIZE
    })

@app.route('/style-transfer', methods=['POST'])
def transfer_style():
    try:
        # Get content and style images and parameters
        if 'content_image' not in request.files or 'style_image' not in request.files:
            return jsonify({"success": False, "error": "Both content_image and style_image are required"}), 400
        
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        
        num_steps = int(request.form.get('num_steps', 300))
        content_weight = float(request.form.get('content_weight', 1.0))
        style_weight = float(request.form.get('style_weight', 1e6))
        learning_rate = float(request.form.get('learning_rate', 0.02))
        
        print(f'\n--- Starting Style Transfer ---')
        print(f'Parameters: steps={num_steps}, content_weight={content_weight}, style_weight={style_weight}')
        
        # Load images
        content_bytes = content_file.read()
        style_bytes = style_file.read()
        
        content_img = load_image_from_bytes(content_bytes)
        style_img = load_image_from_bytes(style_bytes)
        
        # Perform style transfer
        result = run_style_transfer(
            content_img, 
            style_img, 
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            learning_rate=learning_rate
        )
        
        # Convert to image and encode
        result_pil = tensor_to_image(result)
        
        buffered = io.BytesIO()
        result_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print('Style transfer completed successfully!\n')
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        print(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False)