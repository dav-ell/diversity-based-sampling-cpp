// main.cpp
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "H5Cpp.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

/**
 * Helper function to get the current time as a formatted string.
 */
std::string current_time() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return oss.str();
}

/**
 * Logging macros.
 */
#define LOG_INFO(msg) std::cout << "[" << current_time() << "][INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[" << current_time() << "][ERROR] " << msg << std::endl

/**
 * Load all embeddings from an HDF5 file.
 * The HDF5 file is expected to contain a dataset with the name provided by embeddings_keyword
 * containing a 2D matrix of floats (rows = embeddings, columns = dimensions).
 *
 * @param file_path Path to the HDF5 file.
 * @param embeddings Pointer reference to store the embeddings (flat array).
 * @param n_items Number of embeddings (rows).
 * @param dim Dimensionality of each embedding.
 * @param embeddings_keyword The name of the dataset to load embeddings from.
 * @return True if successful, false otherwise.
 */
bool load_embeddings_hdf5(const string& file_path, float*& embeddings, int& n_items, int& dim, const string& embeddings_keyword) {
    try {
        LOG_INFO("Opening HDF5 file: " << file_path);
        // Open the HDF5 file in read-only mode.
        H5::H5File file(file_path, H5F_ACC_RDONLY);
        // Open the dataset with the given embeddings keyword.
        LOG_INFO("Opening dataset '" << embeddings_keyword << "'.");
        H5::DataSet dataset = file.openDataSet(embeddings_keyword);
        // Get the dataspace and verify its dimensionality.
        H5::DataSpace dataspace = dataset.getSpace();
        int ndims = dataspace.getSimpleExtentNdims();
        if (ndims != 2) {
            LOG_ERROR("Dataset '" << embeddings_keyword << "' is not 2-dimensional.");
            return false;
        }
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        n_items = static_cast<int>(dims[0]);
        dim = static_cast<int>(dims[1]);
        LOG_INFO("Dataset dimensions: " << n_items << " x " << dim);

        // Allocate memory for the embeddings using memory mapping.
        size_t total_size = static_cast<size_t>(n_items) * dim * sizeof(float);
        LOG_INFO("Allocating memory for embeddings (" << total_size << " bytes) using mmap.");
        embeddings = static_cast<float*>(mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (embeddings == MAP_FAILED) {
            LOG_ERROR("Memory mapping failed for embeddings.");
            return false;
        }

        // Read the entire dataset into the memory-mapped array.
        LOG_INFO("Reading dataset into memory.");
        dataset.read(embeddings, H5::PredType::NATIVE_FLOAT);
        LOG_INFO("Successfully loaded embeddings.");
    } catch (H5::Exception& error) {
        LOG_ERROR("Error reading HDF5 file: " << error.getDetailMsg());
        return false;
    }
    return true;
}

/**
 * Read the "image_paths" attribute from the HDF5 file.
 * The attribute is expected to be an array of strings saved by the Python script.
 *
 * @param file_path Path to the HDF5 file.
 * @return A vector of image path strings.
 */
vector<string> read_image_paths(const string& file_path) {
    vector<string> imagePaths;
    try {
        LOG_INFO("Opening HDF5 file to read image paths attribute: " << file_path);
        H5::H5File file(file_path, H5F_ACC_RDONLY);
        if (!file.attrExists("image_paths")) {
            throw std::runtime_error("Attribute 'image_paths' not found in the file.");
        }
        H5::Attribute attr = file.openAttribute("image_paths");
        H5::DataType type = attr.getDataType();
        H5::DataSpace space = attr.getSpace();
        hsize_t n_strings = 0;
        space.getSimpleExtentDims(&n_strings, NULL);

        // Prepare buffer to read variable-length strings.
        char** rdata = new char*[n_strings];
        attr.read(type, rdata);

        for (hsize_t i = 0; i < n_strings; ++i) {
            if (rdata[i])
                imagePaths.push_back(string(rdata[i]));
            else
                imagePaths.push_back("");
        }

        // Free memory allocated by HDF5 for variable-length strings.
        for (hsize_t i = 0; i < n_strings; ++i) {
            free(rdata[i]);
        }
        delete[] rdata;
        LOG_INFO("Successfully read " << imagePaths.size() << " image paths from attribute.");
    } catch (H5::Exception& error) {
        LOG_ERROR("Error reading image_paths attribute: " << error.getDetailMsg());
        throw;
    }
    return imagePaths;
}

/**
 * LSH-based diversity sampling using random hyperplanes (for Angular distance).
 * 
 * This function computes k hyperplanes where k = num_to_sample.
 * For each embedding, a hash is computed as a bitwise combination of the sign
 * of the dot products with each hyperplane. Embeddings are bucketized by the 
 * computed hash, and one embedding per bucket is selected. If buckets do not 
 * provide enough samples, random embeddings are added until num_to_sample is reached.
 *
 * @param embeddings Flat array of embeddings.
 * @param n_items Number of embeddings.
 * @param dim Dimensionality of embeddings.
 * @param num_to_sample Number of embeddings to sample.
 * @param metric The distance metric ("angular" supported).
 * @return Pointer to array of selected indices. The caller must free the memory using munmap.
 */
int* diversity_sample_lsh(const float* embeddings, int n_items, int dim, int num_to_sample, const string& metric) {
    if (metric != "angular") {
        LOG_ERROR("Only 'angular' metric is supported in LSH-based sampling.");
        return nullptr;
    }

    LOG_INFO("Starting LSH-based diversity sampling with " << num_to_sample << " samples out of " << n_items << " embeddings using metric: " << metric);

    // Determine the number of hyperplanes.
    int k = num_to_sample;
    LOG_INFO("Generating " << k << " hyperplanes for LSH hashing.");

    // Generate k hyperplanes with normally distributed random values.
    std::vector<std::vector<float>> hyperplanes(k, std::vector<float>(dim));
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dim; ++d) {
            hyperplanes[i][d] = norm_dist(rng);
        }
    }

    // Bucketize embeddings based on computed LSH hash.
    std::unordered_map<size_t, std::vector<int>> buckets;
    for (int i = 0; i < n_items; ++i) {
        size_t hash = 0;
        const float* emb = embeddings + i * dim;
        for (int j = 0; j < k; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += emb[d] * hyperplanes[j][d];
            }
            // Set bit j if dot product is positive.
            if (dot > 0)
                hash |= (static_cast<size_t>(1) << j);
        }
        buckets[hash].push_back(i);
    }
    LOG_INFO("Bucketization complete. Number of buckets: " << buckets.size());

    // Collect one embedding from each bucket.
    std::vector<int> selected;
    for (const auto& kv : buckets) {
        if (selected.size() >= static_cast<size_t>(num_to_sample))
            break;
        selected.push_back(kv.second[0]);
    }

    // If not enough selected, fill remaining slots with random embeddings.
    if (selected.size() < static_cast<size_t>(num_to_sample)) {
        std::vector<int> all_ids(n_items);
        for (int i = 0; i < n_items; ++i)
            all_ids[i] = i;
        std::shuffle(all_ids.begin(), all_ids.end(), rng);
        for (int id : all_ids) {
            if (selected.size() >= static_cast<size_t>(num_to_sample))
                break;
            if (std::find(selected.begin(), selected.end(), id) == selected.end())
                selected.push_back(id);
        }
    }

    // Allocate memory for selected indices using memory mapping.
    size_t selected_size = static_cast<size_t>(num_to_sample) * sizeof(int);
    int* selected_ids = static_cast<int*>(mmap(nullptr, selected_size, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (selected_ids == MAP_FAILED) {
        LOG_ERROR("Memory mapping failed for selected_ids in LSH sampling.");
        return nullptr;
    }

    // Copy selected indices into the memory-mapped array.
    for (int i = 0; i < num_to_sample; ++i) {
        selected_ids[i] = selected[i];
    }

    LOG_INFO("LSH-based diversity sampling completed. Selected " << num_to_sample << " embeddings.");
    return selected_ids;
}

/**
 * Save a set of embeddings and corresponding image paths into an HDF5 file.
 * The embeddings are saved into a dataset named "embeddings" and the image paths
 * are saved into a dataset named "image_paths".
 *
 * @param file_path Path to the output HDF5 file.
 * @param embeddings Flat array of embeddings to save.
 * @param imagePaths Vector of image path strings corresponding to the embeddings.
 * @param n_items Number of embeddings.
 * @param dim Dimensionality of each embedding.
 * @return True if successful, false otherwise.
 */
bool save_embeddings_and_paths_hdf5(const string& file_path, const float* embeddings,
                                    const vector<string>& imagePaths, int n_items, int dim) {
    hsize_t dims[2] = { static_cast<hsize_t>(n_items), static_cast<hsize_t>(dim) };
    try {
        LOG_INFO("Creating HDF5 file: " << file_path);
        // Create a new HDF5 file (overwrite if it exists).
        H5::H5File file(file_path, H5F_ACC_TRUNC);
        H5::DataSpace dataspace(2, dims);
        // Create the dataset "embeddings".
        LOG_INFO("Creating dataset 'embeddings' with dimensions: " << n_items << " x " << dim);
        H5::DataSet dataset = file.createDataSet("embeddings", H5::PredType::NATIVE_FLOAT, dataspace);
        // Write the data into the dataset.
        dataset.write(embeddings, H5::PredType::NATIVE_FLOAT);
        LOG_INFO("Successfully wrote embeddings to file.");

        // Save the image paths as a dataset.
        hsize_t str_dims[1] = { static_cast<hsize_t>(imagePaths.size()) };
        H5::DataSpace str_dataspace(1, str_dims);
        // Create a variable-length string type.
        H5::StrType strdatatype(H5::PredType::C_S1, H5T_VARIABLE);
        LOG_INFO("Creating dataset 'image_paths' with " << imagePaths.size() << " entries.");
        H5::DataSet str_dataset = file.createDataSet("image_paths", strdatatype, str_dataspace);

        // Prepare an array of C-style string pointers.
        vector<const char*> cstrs;
        cstrs.reserve(imagePaths.size());
        for (const auto& s : imagePaths) {
            cstrs.push_back(s.c_str());
        }
        str_dataset.write(cstrs.data(), strdatatype);
        LOG_INFO("Successfully wrote image paths to file.");
    } catch (H5::Exception& error) {
        LOG_ERROR("Error writing HDF5 file: " << error.getDetailMsg());
        return false;
    }
    return true;
}

/**
 * Main function.
 * 
 * Usage: <executable> <hdf5_input_path> <num_to_sample> <hdf5_output_path> <metric> <embeddings_keyword>
 *
 * The metric should be "angular" for LSH-based sampling.
 * The embeddings_keyword argument is the dataset name in the HDF5 file that contains the embeddings.
 */
int main(int argc, char** argv) {
    LOG_INFO("Program started.");

    if (argc != 6) {
        LOG_ERROR("Incorrect number of arguments.");
        std::cerr << "Usage: " << argv[0] << " <hdf5_input_path> <num_to_sample> <hdf5_output_path> <metric> <embeddings_keyword>" << std::endl;
        return 1;
    }

    string input_file = argv[1];
    int num_to_sample = 0;
    try {
        num_to_sample = std::stoi(argv[2]);
    } catch (const std::exception& ex) {
        LOG_ERROR("Invalid number for num_to_sample: " << ex.what());
        return 1;
    }
    string output_file = argv[3];
    string metric = argv[4];
    string embeddings_keyword = argv[5];

    LOG_INFO("Input file: " << input_file);
    LOG_INFO("Number to sample: " << num_to_sample);
    LOG_INFO("Output file: " << output_file);
    LOG_INFO("Distance metric: " << metric);
    LOG_INFO("Embeddings dataset keyword: " << embeddings_keyword);

    // Load embeddings from the HDF5 file using the provided embeddings keyword.
    float* embeddings = nullptr;
    int n_items = 0, dim = 0;
    if (!load_embeddings_hdf5(input_file, embeddings, n_items, dim, embeddings_keyword)) {
        LOG_ERROR("Error loading embeddings from file: " << input_file);
        return 1;
    }
    LOG_INFO("Loaded " << n_items << " embeddings from the file.");

    // Read the image paths from the input HDF5 file attribute.
    vector<string> allImagePaths;
    try {
        allImagePaths = read_image_paths(input_file);
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to read image paths: " << ex.what());
        size_t total_size = static_cast<size_t>(n_items) * dim * sizeof(float);
        munmap(embeddings, total_size);
        return 1;
    }
    if (allImagePaths.size() != static_cast<size_t>(n_items)) {
        LOG_ERROR("Mismatch between number of embeddings and image paths.");
        size_t total_size = static_cast<size_t>(n_items) * dim * sizeof(float);
        munmap(embeddings, total_size);
        return 1;
    }

    // Perform LSH-based diversity sampling.
    int* selected_ids = diversity_sample_lsh(embeddings, n_items, dim, num_to_sample, metric);
    if (!selected_ids) {
        LOG_ERROR("LSH-based diversity sampling failed.");
        size_t total_size = static_cast<size_t>(n_items) * dim * sizeof(float);
        munmap(embeddings, total_size);
        return 1;
    }
    LOG_INFO("Selected " << num_to_sample << " diverse embeddings.");

    // Allocate memory for the selected embeddings.
    size_t selected_emb_size = static_cast<size_t>(num_to_sample) * dim * sizeof(float);
    LOG_INFO("Allocating memory for selected embeddings (" << selected_emb_size << " bytes) using mmap.");
    float* selected_embeddings = static_cast<float*>(mmap(nullptr, selected_emb_size, PROT_READ | PROT_WRITE,
                                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (selected_embeddings == MAP_FAILED) {
        LOG_ERROR("Memory mapping failed for selected_embeddings.");
        munmap(embeddings, static_cast<size_t>(n_items) * dim * sizeof(float));
        munmap(selected_ids, static_cast<size_t>(num_to_sample) * sizeof(int));
        return 1;
    }

    // Collect the selected embeddings and corresponding image paths.
    vector<string> selectedImagePaths;
    LOG_INFO("Collecting selected embeddings and image paths into contiguous memory.");
    for (int i = 0; i < num_to_sample; ++i) {
        int idx = selected_ids[i];
        if (idx < 0 || idx >= n_items) {
            LOG_ERROR("Invalid selected index: " << idx);
            continue;
        }
        const float* src_ptr = embeddings + idx * dim;
        float* dst_ptr = selected_embeddings + i * dim;
        std::copy(src_ptr, src_ptr + dim, dst_ptr);
        selectedImagePaths.push_back(allImagePaths[idx]);
    }

    // Save the selected embeddings and image paths to a new HDF5 file.
    if (!save_embeddings_and_paths_hdf5(output_file, selected_embeddings, selectedImagePaths, num_to_sample, dim)) {
        LOG_ERROR("Failed to save selected embeddings and image paths to: " << output_file);
        munmap(embeddings, static_cast<size_t>(n_items) * dim * sizeof(float));
        munmap(selected_ids, static_cast<size_t>(num_to_sample) * sizeof(int));
        munmap(selected_embeddings, selected_emb_size);
        return 1;
    }

    LOG_INFO("Successfully saved selected embeddings and image paths to: " << output_file);

    // Cleanup all memory-mapped regions.
    munmap(embeddings, static_cast<size_t>(n_items) * dim * sizeof(float));
    munmap(selected_ids, static_cast<size_t>(num_to_sample) * sizeof(int));
    munmap(selected_embeddings, selected_emb_size);

    LOG_INFO("Program completed successfully.");
    return 0;
}