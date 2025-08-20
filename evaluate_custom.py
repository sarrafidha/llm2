import json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Chemins à adapter
image_path = "C:/Users/sarra/Downloads/changechat/train/train/image/im1/00004.png"
question = "What is the percentage of unchanged areas?"

# Charger le processor et le modèle LLaVA (ici version 7B, existe aussi en 13B)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to("cuda" if torch.cuda.is_available() else "cpu")

# Charger l'image
image = Image.open(image_path).convert("RGB")

# Préparer l'entrée
prompt = f"USER: <image>\n{question}\nASSISTANT:"
inputs = processor(prompt, image, return_tensors="pt").to(model.device)

# Générer la réponse
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(output[0], skip_special_tokens=True)

print("Question :", question)
print("Réponse LLaVA :", answer) 