# openclip.py

import os
import fire
import numpy as np
import h5py
from PIL import Image
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.cuda.amp as amp

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            return self.preprocess(img)

def convert_images_to_hdf5(input_dir, output_file, batch_size, num_workers=4):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize OpenCLIP
    clip_embd = OpenCLIPEmbeddings(
        model_name="ViT-H-14-378-quickgelu",
        checkpoint="dfn5b"
    )
    clip_embd.model = clip_embd.model.to(device)
    
    # Collect image paths
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, clip_embd.preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False
    )
    
    # Pre-allocate embeddings array
    embedding_dim = clip_embd.model.visual.output_dim
    embeddings = np.zeros((len(image_paths), embedding_dim), dtype=np.float32)
    
    # Processing with AMP
    scaler = amp.GradScaler('cuda') if device.type == "cuda" else None  # Updated for PyTorch 2.4+
    
    with h5py.File(output_file, 'w') as hf:
        hf.attrs['model_name'] = clip_embd.model_name
        hf.attrs['checkpoint'] = clip_embd.checkpoint
        dt = h5py.string_dtype(encoding='utf-8')
        hf.attrs.create('image_paths', np.array(image_paths, dtype=dt))
        
        offset = 0
        for batch_tensors in tqdm(dataloader, desc="Processing images"):
            batch_size_current = len(batch_tensors)
            
            # Move preprocessed tensors to device
            batch_tensors = batch_tensors.to(device)
            
            # Generate embeddings with AMP
            with torch.no_grad():
                if scaler is not None:
                    with amp.autocast():
                        batch_embeddings = clip_embd.model.encode_image(batch_tensors)
                else:
                    batch_embeddings = clip_embd.model.encode_image(batch_tensors)
                    
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Store embeddings
            embeddings[offset:offset + batch_size_current] = batch_embeddings
            offset += batch_size_current
            
            # Clear GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        hf.create_dataset('embeddings', data=embeddings)
    
    return embeddings

def main(input_dir, output_file, batch_size=32, num_workers=4):
    convert_images_to_hdf5(input_dir, output_file, batch_size, num_workers)

if __name__ == '__main__':
    fire.Fire(main)