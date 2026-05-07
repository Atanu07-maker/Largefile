import os
import sys
from types import ModuleType

# Fix basicsr bug: it looks for a module that moved in newer torchvision
try:
    import torchvision.transforms.functional_tensor as T
except ImportError:
    # Create a fake module to satisfy the import
    mock_module = ModuleType("torchvision.transforms.functional_tensor")
    sys.modules["torchvision.transforms.functional_tensor"] = mock_module
    import torchvision.transforms.functional as F
    # Map the functions basicsr expects
    mock_module.rgb_to_grayscale = F.rgb_to_grayscale

import streamlit as st
import cv2
# ... (rest of your imports)
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
from gfpgan import GFPGANer

st.set_page_config(page_title="Pro FaceSwap", layout="centered")

# --- UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models (Cached) ---
@st.cache_resource
def load_all_models():
    # Use 'buffalo_s' for lower RAM usage
    app = FaceAnalysis(name='buffalo_s') 
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Ensure the .onnx file is in your GitHub repo root!
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)
    
    restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', upscale=1)
    return app, swapper, restorer

app, swapper, restorer = load_all_models()

# --- App Layout ---
st.title("🎭 Pro-Level Face Swapper")
st.info("Tip: Use clear, front-facing photos for the best results.")

col1, col2 = st.columns(2)
with col1:
    target_file = st.file_uploader("Target Image (Body)", type=['jpg', 'png', 'jpeg'])
with col2:
    source_file = st.file_uploader("Source Face", type=['jpg', 'png', 'jpeg'])

enhance = st.checkbox("Enable High-Definition Restoration (Recommended)", value=True)

if target_file and source_file:
    # Convert to OpenCV
    target_img = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)
    source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if st.button("Generate Swapped Image"):
        with st.spinner("Analyzing and enhancing..."):
            # 1. Detection
            t_faces = app.get(target_img)
            s_faces = app.get(source_img)

            if not t_faces or not s_faces:
                st.error("Face not detected! Try an image where the person is looking at the camera.")
            else:
                # 2. Perform Swap
                # Sort by size to pick the largest face (the main subject)
                t_face = sorted(t_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
                s_face = sorted(s_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
                
                result = swapper.get(target_img, t_face, s_face, paste_back=True)

                # 3. Enhance (The 'Perfect' Step)
                if enhance:
                    # GFPGAN cleans up the blurry output of the swapper
                    _, _, result = restorer.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)

                # 4. Display Result
                res_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="Enhanced Outcome", use_container_width=True)

                # Download button
                res_pil = Image.fromarray(res_rgb)
                res_pil.save("final_output.jpg", quality=95)
                with open("final_output.jpg", "rb") as f:
                    st.download_button("Download High-Res Image", f, "faceswap_pro.jpg", "image/jpeg")
