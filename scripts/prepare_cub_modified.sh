%%bash
set -e  # Exit if any command fails

# Define the root directory
CUB_ROOT='resource/datasets/CUB_200_2011/'

# Check if the dataset directory exists and proceed with the remaining commands
if [[ ! -d "${CUB_ROOT}" ]]; then
    # Assume the file is already downloaded, just extract it
    echo "Extracting the CUB_200_2011 dataset..."
    tar -zxf resource/datasets/CUB_200_2011.tgz -C resource/datasets
    
    # Ensure extraction is successful
    if [[ $? -eq 0 ]]; then
        echo "Dataset extracted successfully."
    else
        echo "Error in extracting dataset."
        exit 1
    fi
fi

# Generate train.txt and test.txt splits using the provided Python script
echo "Generating the train.txt/test.txt split files"
python3 scripts/split_cub_for_cbml_loss.py