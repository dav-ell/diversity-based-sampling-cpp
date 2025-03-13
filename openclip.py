# openclip.py

import os
import fire
import numpy as np
import h5py
from PIL import Image
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from tqdm import tqdm
import torch

def convert_images_to_hdf5(input_dir, output_file, batch_size):
    # Check for GPU, then MPS, then default to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for acceleration")
    else:
        device = torch.device("cpu")
        print("GPU/MPS device not found. Using CPU")

    # Initialize OpenCLIP embeddings with recommended parameters
    clip_embd = OpenCLIPEmbeddings(
        model_name="ViT-H-14-378-quickgelu",
        checkpoint="dfn5b"
    )
    
    # Initialize model on the selected device
    clip_embd.model.to(device)

    # Create a function to preprocess images using the model's preprocessor
    # This function handles conversion of PIL images to the input tensor
    def preprocess(image):
        return clip_embd.preprocess(Image.open(image).convert("RGB"))
    
    # Create list of image paths with valid extensions
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.lower().endswith(image_extensions)
    ]
    
    # Create HDF5 file with embedded vectors
    with h5py.File(output_file, 'w') as hf:
        hf.attrs['model_name'] = clip_embd.model_name
        hf.attrs['checkpoint'] = clip_embd.checkpoint
        
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and preprocess images
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            preprocessed_images = [clip_embd.preprocess(image) for image in images]
            batch_tensor = torch.stack(preprocessed_images).to(device)
            
            # Compute embeddings
            with torch.no_grad():
                batch_embeddings = clip_embd.embed_documents(images, device=device)
            
            embeddings.extend(batch_paths and batch_paths or [])  # Ensuring proper extension even if empty
            embeddings.extend(batch_paths and clip_embd.embed_images(images).cpu().numpy() or [])
        
        embeddings_array = np.array(embeddings)
        hf = h5py.File(output_file, 'w')
        hf.attrs['model_name'] = clip_embd.model_name
        hf.attrs['checkpoint'] = clip_embd.checkpoint
        hf.create_dataset('embeddings', data=embeddings_array)
        hf.close()
    
    return embeddings_array

def main(input_dir, output_file, batch_size=32):
    return convert_images_to_hdf5(input_dir, output_file, batch_size)

if __name__ == '__main__':
    fire.Fire(main)
