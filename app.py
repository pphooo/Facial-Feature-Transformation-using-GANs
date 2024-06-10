import os
from pyngrok import ngrok
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace

from utils.common import tensor2im
from models.psp import pSp
from editings import latent_editor

# Start ngrok tunnel
public_url = ngrok.connect(port='8501')
print(f"Streamlit Public URL: {public_url}")

# Setup model and load pre-trained weights
CODE_DIR = 'encoder4editing'
os.chdir(f'./{CODE_DIR}')

def download_model():
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    
    file_id = "1cUv_reLE6k3604or78EranS7XzuVMWeO"
    file_name = "e4e_ffhq_encode.pt"
    file_dst = os.path.join("pretrained_models", file_name)
    
    if not os.path.exists(file_dst):
        os.makedirs("pretrained_models", exist_ok=True)
        downloaded = drive.CreateFile({'id': file_id})
        downloaded.FetchMetadata(fetch_all=True)
        downloaded.GetContentFile(file_dst)
    else:
        print(f'{file_name} already exists!')

download_model()

# Load model
experiment_type = 'ffhq_encode'
model_path = "pretrained_models/e4e_ffhq_encode.pt"
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()

# Define transformations
EXPERIMENT_ARGS = {
    "ffhq_encode": {
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
}
img_transforms = EXPERIMENT_ARGS[experiment_type]['transform']
resize_dims = (256, 256)

# Define function for alignment
def run_alignment(image_path):
    import dlib
    from utils.alignment import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image

def main():
    st.sidebar.title("About")

    st.sidebar.info("""
    Machine Learning for Facial-Feature-Transformation-using-GANs 
    By Phoowara Watchararat
    """)
    
    st.title('Facial-Feature-Transformation-using-GANs')
    uploaded_file = st.file_uploader("Choose a video...", type=["jpg", "png"])
  
    if uploaded_file is not None:
        upload_name = "playback/temp_video.mp4"
        
        image = Image.open(uploaded_file).convert("RGB")
        image_path = "/content/uploaded_image.jpg"
        image.save(image_path)
    
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
        if experiment_type == "ffhq_encode":
            if 'shape_predictor_68_face_landmarks.dat' not in os.listdir():
                !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                !bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
        
            aligned_image = run_alignment(image_path)
            aligned_image.resize(resize_dims)
            transformed_image = img_transforms(aligned_image)
        
            with torch.no_grad():
                tic = time.time()
                images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
                result_image, latent = images[0], latents[0]
                toc = time.time()
                st.write('Inference took {:.4f} seconds.'.format(toc - tic))
            
                result_image = tensor2im(result_image)
                st.image(result_image, caption='Result Image', use_column_width=True)
            
                is_cars = experiment_type == 'cars_encode'
                editor = latent_editor.LatentEditor(net.decoder, is_cars)
                edited_image = editor.apply_sefa(latents, start_distance=20, step=1).resize((256, 256))
                st.image(edited_image, caption='Edited Image', use_column_width=True)
    else:
        st.subheader("Please upload a image file.")

if __name__ == '__main__':
    main()
