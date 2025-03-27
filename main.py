from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from PIL import Image
import torch
import io
import os
from model import MyCIFARModel

print("🔥 FastAPI app is starting...")

app = FastAPI()

print("📁 Mounting static files...")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    print("📄 Serving index.html")
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

print("🧠 Initializing model...")
model = MyCIFARModel()
try:
    print("📦 Loading model weights...")
    model.load_state_dict(torch.load("cifar10_model.pth", map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]
        return JSONResponse(content={"prediction": label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
