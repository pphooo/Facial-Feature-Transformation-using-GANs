import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import streamlit as st

# ตั้งค่า directory
CODE_DIR = 'encoder4editing'
if not os.path.exists(CODE_DIR):
    os.system(f'git clone https://github.com/omertov/encoder4editing.git {CODE_DIR}')
os.chdir(CODE_DIR)
os.makedirs("pretrained_models", exist_ok=True)

# ดาวน์โหลดโมเดลโดยใช้ gdown
import gdown
MODEL_PATHS = {
    "ffhq_encode": {"id": "1cUv_reLE6k3604or78EranS7XzuVMWeO", "name": "e4e_ffhq_encode.pt"}
}
model_path = MODEL_PATHS["ffhq_encode"]
model_dst = f"pretrained_models/{model_path['name']}"
if not os.path.exists(model_dst):
    gdown.download(id=model_path["id"], output=model_dst, quiet=False)

# โหลดโมเดล
from argparse import Namespace
from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

ckpt = torch.load(model_dst, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_dst
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

# ตั้งค่า Streamlit app
st.title('Image Encoder using e4e')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    # การ alignment
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')

    import dlib
    from utils.alignment import align_face

    def run_alignment(image_path):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(image_path, predictor)
        return aligned_image

    input_image = run_alignment(image)
    st.image(input_image, caption='Aligned Image', use_column_width=True)

    # การแปลงภาพ
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = transform(input_image)

    # รันโมเดล
    def run_on_batch(inputs, net):
        images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        return images, latents

    with torch.no_grad():
        tic = time.time()
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
        toc = time.time()
        st.write('Inference took {:.4f} seconds.'.format(toc - tic))

        # แปลง tensor เป็นภาพ
        result_image = tensor2im(result_image)
        result_image = Image.fromarray(result_image)
        
        st.image(result_image, caption='Result Image', use_column_width=True)

    # เพิ่มฟังก์ชันการแก้ไขเพิ่มเติมตามที่ต้องการ
