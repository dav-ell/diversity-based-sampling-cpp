# Diversity Sampling and Image Extraction System

This repository contains a complete system for generating image embeddings using OpenCLIP, performing LSH-based diversity sampling on these embeddings, and then copying the corresponding images into a designated folder. The overall flow is as follows:

1. **Acquire a Directory of Images:**  
   Prepare a directory containing the images you want to process. Supported image formats are PNG, JPG, and JPEG.

2. **Generate Embeddings with OpenCLIP (openclip.py):**  
   Run the Python script to convert images into embeddings and save them, along with image paths, into an HDF5 file. This script supports CUDA, MPS, and CPU devices.

3. **Diversity Sampling (main.cpp):**  
   Build and run the C++ application which reads the HDF5 file, performs LSH-based diversity sampling on the embeddings, and saves the selected embeddings along with their image paths into a new HDF5 file.  
   **Note:** You must specify the dataset name (embeddings keyword) used to store the embeddings in the HDF5 file.

4. **Copy Selected Images (copy_selected_images.cpp):**  
   Build and run the C++ program to read the selected image paths from the HDF5 file and copy these images to a destination folder.

---

## Dependencies

### Python
- Python 3.7 or higher
- [fire](https://pypi.org/project/fire/)
- [numpy](https://pypi.org/project/numpy/)
- [h5py](https://pypi.org/project/h5py/)
- [Pillow](https://pypi.org/project/Pillow/)
- [torch](https://pypi.org/project/torch/)
- [tqdm](https://pypi.org/project/tqdm/)
- [langchain_experimental.open_clip](https://github.com/huggingface/open_clip) (or your specific OpenCLIP embeddings package)

### C++
- A C++ compiler that supports C++11 (or later)
- HDF5 C++ libraries installed (headers and libraries)
- POSIX system (for mmap and filesystem operations)

---

## Build Instructions

### C++ Components

#### 1. Build `main.cpp`
```bash
g++ -std=c++11 main.cpp -o diversity_sampler -lhdf5_cpp -lhdf5
```

Make sure that the HDF5 libraries are in your library path. You may need to adjust the include/library paths using `-I` and `-L` options if they are not in standard locations.

#### 2. Build `copy_selected_images.cpp`
```bash
g++ -std=c++17 copy_selected_images.cpp -o copy_images -lhdf5_cpp -lhdf5
```
The above command uses C++17 for filesystem support. Adjust include/library paths if necessary.

---

## Usage

### Step 1: Generate Image Embeddings with OpenCLIP

Run the Python script to create an HDF5 file containing image embeddings and image paths.
```bash
python openclip.py --input_dir /path/to/images --output_file embeddings.h5 --batch_size 32
```
- `--input_dir`: Path to the directory containing your images.
- `--output_file`: Name (and path) of the HDF5 file to be created.
- `--batch_size`: (Optional) Batch size for processing images. Default is 32.

### Step 2: Run Diversity Sampling (main.cpp)

After generating the HDF5 file, run the diversity sampling program. Note that you must provide the embeddings keyword used in the HDF5 file (by default, the Python script stores the embeddings in a dataset named `embeddings`).
```bash
./diversity_sampler /path/to/embeddings.h5 100 /path/to/selected_embeddings.h5 angular embeddings
```
Parameters:
- `/path/to/embeddings.h5`: Path to the input HDF5 file.
- `100`: Number of embeddings to sample.
- `/path/to/selected_embeddings.h5`: Path for the output HDF5 file.
- `angular`: Distance metric (currently only “angular” is supported).
- `embeddings`: The keyword (dataset name) in the HDF5 file where the embeddings are stored.

### Step 3: Copy Selected Images (copy_selected_images.cpp)

Finally, run the program to copy the selected images to a destination folder.
```bash
./copy_images /path/to/selected_embeddings.h5 /path/to/destination_folder
```
Parameters:
- `/path/to/selected_embeddings.h5`: HDF5 file containing the selected embeddings and image paths.
- `/path/to/destination_folder`: Destination folder where the images will be copied. The folder will be created if it does not exist.

---

## Notes
- **Error Handling:** Both C++ applications include robust argument parsing and error logging. If an error occurs (e.g., missing file, invalid arguments), detailed error messages will be printed to help diagnose the issue.
- **Memory Management:** The C++ programs use memory mapping (mmap) for efficient handling of large arrays. Ensure that your system has enough virtual memory to support the operations.
- **Device Support for OpenCLIP:** The `openclip.py` script automatically selects the best available device in the order of CUDA, MPS, and CPU.

---

## Contact

For issues or contributions, please open an issue or submit a pull request on GitHub.

