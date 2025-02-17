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
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not found. Using CPU")

    # Initialize OpenCLIP embeddings with recommended parameters
    clip_embd = OpenCLIPEmbeddings(
        model_name="ViT-H-14-378-quickgelu",
        checkpoint="dfn5b"
    )
    
    # Move the model to the appropriate device
    clip_embd.model = clip_embd.model.to(device)
    
    # Collect image paths with valid extensions
    image_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Create HDF5 file with embedded vectors
    with h5py.File(output_file, 'w') as hf:
        hf.attrs['model_name'] = clip_embd.model_name
        hf.attrs['checkpoint'] = clip_embd.checkpoint
        
        dt = h5py.string_dtype(encoding='utf-8')
        hf.attrs.create('image_paths', np.array(image_paths, dtype=dt))
        
        embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images using PIL and convert to RGB
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # Preprocess each image and stack them into a batch tensor
            preprocessed_images = torch.stack([clip_embd.preprocess(image) for image in images]).to(device)
            
            # Generate embeddings for the current batch
            with torch.no_grad():
                batch_embeddings_tensor = clip_embd.model.encode_image(preprocessed_images)
                # Move embeddings to CPU for numpy conversion
                batch_embeddings = batch_embeddings_tensor.cpu().numpy()
            
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings)
        hf.create_dataset('embeddings', data=embeddings_array)
    
    return embeddings_array

def main(input_dir, output_file, batch_size=32):
    convert_images_to_hdf5(input_dir, output_file, batch_size)

if __name__ == '__main__':
    fire.Fire(main)