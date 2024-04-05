# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
###########################
###########################
###########################
###########################
# ALSO ADD MODELS, CHECKPOINTS, ECT. REQUIRED 
# ALSO ADD CUSTOM NODES INSTALLS
###########################
# MAKE THIS REPO: https://github.com/Aiaesthetica/ComfyUI.git PUBLIC BEFORE RUNNING THIS !!!!!!!!!
RUN git clone https://github.com/Aiaesthetica/ComfyUI.git /comfyui


# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt

# Install runpod
RUN pip3 install runpod requests
# Download all models that workflow requires
# checkpoints
RUN wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
RUN wget -U Mozilla/5.0 -O models/checkpoints/pornmasterPro_proDPOV1.safetensors https://civitai.com/api/download/models/340218
# VAEs
RUN wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
RUN wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
RUN wget -O models/vae/vae-ft-mse-840000-ema-pruned.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true
# clip_vision
RUN wget -O models/clip_vision/clip_vision.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true
# loras
RUN wget -O models/loras/xl_more_art-full_v1.safetensors https://civitai.com/api/download/models/152309
RUN wget -U mozilla -O models/loras/GodPussy1v4.safetensors https://civitai.com/api/download/models/99602
RUN wget -O models/loras/ip-adapter-faceid_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors?download=true
RUN wget -O models/loras/ip-adapter-faceid-plus_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors?download=true
RUN wget -O models/loras/ip-adapter-full-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors
RUN wget -O models/loras/ip-adapter-plus-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors
RUN wget -O models/loras/Masturbation with Dildo v1.1.safetensors https://civitai.com/api/download/models/269151?type=Model&format=SafeTensor
RUN wget -O models/loras/Realistic_Visionary_NSFW-07.safetensors https://civitai.com/api/download/models/199856?type=Model&format=SafeTensor

# insightface
RUN wget -O models/insightface/models/antelopev2/1k3d68.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx?download=true
RUN wget -O models/insightface/models/antelopev2/2d106det.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx?download=true
RUN wget -O models/insightface/models/antelopev2/genderage.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx?download=true
RUN wget -O models/insightface/models/antelopev2/glintr100.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx?download=true
RUN wget -O models/insightface/models/antelopev2/scfd_10g_bnkps.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx?download=true
RUN wget -O models/insightface/models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true
# controlnet
RUN wget -O models/controlnet/diffusion_pytorch_model.safetensors https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true
# ipadapter
RUN wget -O models/ipadapter/ip-adapter-faceid_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin
RUN wget -O models/ipadapter/ip-adapter-faceid-plus_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin
RUN wget -O models/ipadapter/ip-adapter-full-face_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.bin?download=true
RUN wget -O models/ipadapter/ip-adapter-full-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors
RUN wget -O models/ipadapter/ip-adapter-plus-face_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin?download=true
RUN wget -O models/ipadapter/ip-adapter-plus-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors

# custom nodes
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git /custom_nodes/ComfyUI_InstantID
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /custom_nodes/ComfyUI_IPAdapter_plus
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/twri/sdxl_prompt_styler.git /custom_nodes/sdxl_prompt_styler
RUN git clone https://github.com/Stability-AI/stability-ComfyUI-nodes.git /custom_nodes/stability-ComfyUI-nodes
RUN git clone git clone https://github.com/city96/ComfyUI_NetDist /custom_nodes/ComfyUI_NetDist 

WORKDIR /comfyui/custom_nodes
RUN pip install requests 
# Example for adding specific models into image
# ADD models/checkpoints/sd_xl_base_1.0.safetensors models/checkpoints/
# ADD models/vae/sdxl_vae.safetensors models/vae/

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
