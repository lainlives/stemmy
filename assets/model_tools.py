import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import HfApi, snapshot_download


def extract_urls_from_file(input_filename, output_filename):
    """
    Extracts all URLs from an input text file and writes them to an output file.
    """
    # A general regular expression for finding URLs
    # It looks for strings starting with http:// or https://, followed by non-whitespace characters
    URL_REGEX = r"https?://\S+|www\.\S+"

    try:
        # 1. Read the contents of the input file
        with open(input_filename, "r", encoding="utf-8") as f_in:
            content = f_in.read()

        # 2. Find all URLs in the content using re.findall()
        urls = re.findall(URL_REGEX, content)

        # Ensure only unique URLs are written by converting the list to a set and back to a list
        unique_urls = sorted(list(set(urls)))

        # 3. Write the extracted URLs to the output file, each on a new line
        with open(output_filename, "w", encoding="utf-8") as f_out:
            for url in unique_urls:
                f_out.write(url + "\n")

        print(
            f"Found {len(unique_urls)} unique URLs and saved them to {output_filename}"
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def remove_chars_from_file(input_filename, chars_to_remove):
    """
    Reads a text file, removes specified characters, and writes the changes back to the file.

    Args:
        input_filename (str): The name of the input text file.
        chars_to_remove (list): A list of characters to be removed (e.g., [',', '"', '}']).
    """
    try:
        # Read the file content
        with open(input_filename, "r") as file:
            content = file.read()

        # Remove the characters
        for char in chars_to_remove:
            content = content.replace(char, "")

        # Write the modified content back to the file
        with open(input_filename, "w") as file:
            file.write(content)

        print(
            f"Successfully removed characters {chars_to_remove} from {input_filename}"
        )

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_file_hash(filepath, block_size=65536):
    """Calculates the SHA256 hash of a file's content."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(block_size):
                sha256.update(chunk)
    except FileNotFoundError:
        return None  # Handle cases where a file might be deleted during the scan

    return sha256.hexdigest()


def find_and_remove_duplicates(directory="."):
    """Finds duplicate files in the given directory and removes the one with the longer filename."""
    hashes_to_files = defaultdict(list)
    files_to_hash = {}

    # Step 1: Hash all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_hash = calculate_file_hash(filepath)
            if file_hash:
                hashes_to_files[file_hash].append(filepath)
                files_to_hash[filepath] = file_hash

    # Step 2: Identify duplicate groups (more than one file per hash)
    duplicates = {h: files for h, files in hashes_to_files.items() if len(files) > 1}

    if not duplicates:
        print("No duplicate files found.")
        return

    # Step 3: Iterate over duplicates, compare filename length, and delete the longer one
    for file_hash, file_list in duplicates.items():
        # Sort files by filename length (ascending). The one to keep is the first item.
        # If lengths are equal, an arbitrary one is kept.
        files_sorted_by_length = sorted(file_list, key=len)
        file_to_keep = files_sorted_by_length[0]
        files_to_delete = files_sorted_by_length[1:]

        print(f"\nDuplicate group (Hash: {file_hash[:10]}...):")
        print(f"  Keeping: {file_to_keep}")
        for file_to_delete in files_to_delete:
            try:
                os.remove(file_to_delete)
                print(f"  Deleted: {file_to_delete} (longer filename)")
            except OSError as e:
                print(f"  Error deleting {file_to_delete}: {e}")


def download_file(url, local_dir):
    """Helper function to download a single file."""
    try:
        # Extract filename from URL (e.g., https://example.com/file.jpg -> file.jpg)
        filename = url.split("/")[-1].split("?")[0] or "downloaded_file"
        save_path = os.path.join(local_dir, filename)

        # Download the file content
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=31457280):
                f"Initiating download: {filename}"
                f.write(chunk)
        return f"Successfully downloaded: {filename}"
    except Exception as e:
        return f"Failed to download {url}: {e}"


def download_files_from_txt(filename, local_dir, max_workers):
    """Main function to read URLs and download them using 20 threads."""
    # Ensure local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Read URLs from the text file
    with open(filename, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Use ThreadPoolExecutor to handle 20 downloads at a time
<<<<<<< HEAD
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
=======
    with ThreadPoolExecutor(max_workers=5) as executor:
>>>>>>> 10bf2414c0c2b13f9b90593a274c8d054211060a
        # Submit all download tasks to the pool
        results = [executor.submit(download_file, url, local_dir) for url in urls]

        # Monitor results as they complete
        for future in results:
            print(future.result())


def download_files_from_txt_aria(filename, local_dir):
    command = [
        "aria2c",
        "--input-file",
        filename,
        "--dir",
        local_dir,
        "-c",  # Continue downloading a partially downloaded file
        "-j",
        "30",  # Set max concurrent downloads (adjust as needed)
        "-x",
        "16",  # Set max connections per server (adjust as needed)
    ]
    print(f"Starting downloads with aria2c in directory: {os.path.abspath(local_dir)}")
    try:
        # Execute the command
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("All downloads finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during aria2c execution: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # os.remove(filename)
        print(f"Downloaded all files: {filename}")


def download_hf_repo(repo_id, local_dir, repo_type, token):
    if not token:
        token = os.getenv("HF_TOKEN")
    """
    Downloads an entire Hugging Face repository to a specified local directory.
    """
    print(f"Downloading {repo_id} to {local_dir}...")

    # Ensure the target directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the snapshot
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
        local_dir_use_symlinks=False,  # Set to False to ensure actual files are moved to local_dir
        repo_type=repo_type,
    )

    print(f"Download complete! Files are located in: {downloaded_path}")
    return downloaded_path


def remove_duplicate_lines(input_file_path, output_file_path):
    """
    Reads lines from input_file_path, removes duplicates, and writes
    unique lines to output_file_path while preserving order.
    """
    try:
        # Use an ordered set to maintain the original file's line order.
        # An easy way to do this in Python 3.7+ is using a dictionary's keys.
        unique_lines_dict = {}
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                # Store line as a dictionary key; duplicates will be ignored
                unique_lines_dict[line] = None

        unique_lines = unique_lines_dict.keys()

        with open(output_file_path, "w") as output_file:
            # Write all unique lines to the new file
            output_file.writelines(unique_lines)

        print(f"Duplicates removed. Unique lines saved to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def push_to_hf(repo_id, repo_type):
    api = HfApi()

    print(f"Uploading current directory to: {repo_id}")

    # Upload everything in the current directory ('.') to the repo root
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Initial model upload",
    )
    print("Upload complete!")


def push_large_folder_to_hf(repo_id, repo_type):
    api = HfApi()
    print(f"Starting large folder upload to: {repo_id}")

    # 3. Use upload_large_folder for resilience and speed
    # This automatically handles multi-threading and local caching for resuming
    api.upload_large_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type=repo_type,
        # Optional: ignore large junk files to save time
        ignore_patterns=[
            ".git/",
            "__pycache__/",
            "*.tmp",
            ".DS_Store",
            "*.cache",
            "*.trash",
        ],
    )

    print(
        "\nUpload complete! Progress was cached locally; if it failed, just run again to resume."
    )


def get_model_hash(model_path):
    """
    Get the hash of a model file
    """
    # print(f"Getting hash for model at {model_path}")
    try:
        with open(model_path, "rb") as f:
            f.seek(
                -10000 * 1024, 2
            )  # Move the file pointer 10MB before the end of the file
            hash_result = hashlib.md5(f.read()).hexdigest()
            # print(f"Hash for {model_path}: {hash_result}")
            return hash_result
    except IOError:
        with open(model_path, "rb") as f:
            hash_result = hashlib.md5(f.read()).hexdigest()
            # print(f"IOError encountered, hash for {model_path}: {hash_result}")
            return hash_result


def download_file_if_missing(url, local_path):
    """
    Download a file from a URL if it doesn't exist locally
    """
    print(f"Checking if {local_path} needs to be downloaded from {url}")
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}")
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {url} to {local_path}")
    else:
        print(f"{local_path} already exists. Skipping download.")


def load_json_data(file_path):
    """
    Load JSON data from a file
    """
    print(f"Loading JSON data from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"Loaded JSON data successfully from {file_path}")
            return data
    except FileNotFoundError:
        print(f"{file_path} not found.")
        sys.exit(1)


def iterate_and_hash(
    directory,
    vr_model_data_url,
    mdx_model_data_url,
    vr_model_data_local_path,
    mdx_model_data_local_path,
):
    """
    Iterate through a directory and hash all model files
    """
    print(f"Iterating through directory {directory} to hash model files")
    model_files = [
        (file, os.path.join(root, file))
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith((".pth", ".onnx", ".pt", ".ckpt"))
    ]

    download_file_if_missing(vr_model_data_url, vr_model_data_local_path)
    download_file_if_missing(mdx_model_data_url, mdx_model_data_local_path)

    vr_model_data = load_json_data(vr_model_data_local_path)
    mdx_model_data = load_json_data(mdx_model_data_local_path)

    combined_model_params = {
        **vr_model_data,
        **mdx_model_data,
    }

    model_info_list = []
    for file, file_path in sorted(model_files):
        file_hash = get_model_hash(file_path)
        model_info = {
            "file": file,
            "hash": file_hash,
            "params": combined_model_params.get(file_hash, "Parameters not found"),
        }
        model_info_list.append(model_info)

    print(f"Writing model info list to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as json_file:
        json.dump(model_info_list, json_file, indent=4)
        print(f"Successfully wrote model info list to {OUTPUT_PATH}")


def sort_links_by_extension(input_file, output_file):
    # Define the custom priority order
    priority = {
        ".json": 0,
        ".yaml": 1,
        ".th": 2,
        ".pth": 3,
        ".ckpt": 4,
        ".onnx": 5,  # Added .onnx (common typo for .onnx or .onx)
    }

    # Handle the specific user request for .onnx
    # Example: Map .onnx to priority 5
    # priority['.onnx'] = 5

    try:
        with open(input_file, "r") as f:
            # Read lines and strip whitespace/newlines
            links = [line.strip() for line in f if line.strip()]

        def sort_key(link):
            # Extract extension (case-insensitive)
            _, ext = os.path.splitext(link.lower())
            # Return priority index; if not in list, place at the end (index 100)
            return priority.get(ext, 100), link

        # Sort the links
        sorted_links = sorted(links, key=sort_key)

        with open(output_file, "w") as f:
            for link in sorted_links:
                f.write(link + "\n")

        print(f"Successfully sorted links into: {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")


# 1. Load the JSON data
def get_links_from_json(file_input):
    try:
        with open(file_input, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: file not found.")
        data = {}

    # 2. Process and Download
    for model_name, links in data.items():
        if not isinstance(links, list) or len(links) == 0:
            continue
