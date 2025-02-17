// copy_selected_images.cpp
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "H5Cpp.h"

namespace fs = std::filesystem;
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
 * Reads the "image_paths" dataset from an HDF5 file.
 * Note: In our updated file, image paths are saved as an attribute
 * in the Python script, but this file stores them as a dataset.
 * However, if you use the main.cpp to re-save the file, it will store
 * image paths as a dataset. This function supports reading from a dataset.
 *
 * @param file_path Path to the HDF5 file.
 * @return A vector of image path strings.
 */
vector<string> read_selected_image_paths(const string& file_path) {
    vector<string> imagePaths;
    try {
        LOG_INFO("Opening HDF5 file: " << file_path);
        H5::H5File file(file_path, H5F_ACC_RDONLY);
        if (!file.exists("image_paths")) {
            throw std::runtime_error("Dataset 'image_paths' not found in the file.");
        }
        H5::DataSet dataset = file.openDataSet("image_paths");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t n_strings = 0;
        dataspace.getSimpleExtentDims(&n_strings, NULL);

        // Define variable-length string type.
        H5::StrType strdatatype(H5::PredType::C_S1, H5T_VARIABLE);
        char** rdata = new char*[n_strings];
        dataset.read(rdata, strdatatype);

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
        LOG_INFO("Successfully read " << imagePaths.size() << " image paths from dataset.");
    } catch (H5::Exception& error) {
        LOG_ERROR("Error reading image_paths dataset: " << error.getDetailMsg());
        throw;
    }
    return imagePaths;
}

/**
 * Main function.
 * 
 * Usage: <executable> <selected_hdf5_file> <destination_folder>
 *
 * This script reads the "image_paths" dataset (or attribute if re-saved)
 * from the provided HDF5 file and copies each image file to the destination folder.
 */
int main(int argc, char** argv) {
    LOG_INFO("Copy Selected Images Program started.");
    if (argc != 3) {
        LOG_ERROR("Usage: " << argv[0] << " <selected_hdf5_file> <destination_folder>");
        return 1;
    }

    string hdf5_file = argv[1];
    string dest_folder = argv[2];

    // Ensure the destination folder exists; if not, create it.
    try {
        if (!fs::exists(dest_folder)) {
            LOG_INFO("Destination folder does not exist. Creating folder: " << dest_folder);
            fs::create_directories(dest_folder);
        }
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Error creating destination folder: " << e.what());
        return 1;
    }

    // Read selected image paths from HDF5 file.
    vector<string> imagePaths;
    try {
        imagePaths = read_selected_image_paths(hdf5_file);
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to read image paths: " << ex.what());
        return 1;
    }

    // Copy each image file to the destination folder.
    int successCount = 0;
    for (const auto& srcPath : imagePaths) {
        try {
            fs::path source(srcPath);
            if (!fs::exists(source)) {
                LOG_ERROR("Source image does not exist: " << srcPath);
                continue;
            }
            fs::path destPath = fs::path(dest_folder) / source.filename();
            fs::copy_file(source, destPath, fs::copy_options::overwrite_existing);
            LOG_INFO("Copied: " << srcPath << " to " << destPath.string());
            ++successCount;
        } catch (const fs::filesystem_error& e) {
            LOG_ERROR("Failed to copy " << srcPath << ": " << e.what());
        }
    }
    LOG_INFO("Finished copying images. Total images copied: " << successCount);
    LOG_INFO("Program completed successfully.");
    return 0;
}