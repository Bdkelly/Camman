import json
import os

def concatenate_json_files(json_file_1_path, json_file_2_path, output_combined_path):
    """
    Concatenates two JSON files, each containing a list of frame annotations and metadata,
    into a single output JSON file. The content of json_file_2 is appended directly
    after json_file_1 without re-indexing frame numbers.

    Args:
        json_file_1_path (str): Path to the first JSON file.
        json_file_2_path (str): Path to the second JSON file.
        output_combined_path (str): Full path to save the concatenated JSON file.
    """
    try:
        with open(json_file_1_path, 'r') as f:
            data_file1 = json.load(f)
            print(f"DEBUG: Loaded {len(data_file1)} entries from {json_file_1_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_1_path}. Cannot concatenate.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_1_path}. Check file format.")
        return

    try:
        with open(json_file_2_path, 'r') as f:
            data_file2 = json.load(f)
            print(f"DEBUG: Loaded {len(data_file2)} entries from {json_file_2_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_2_path}. Cannot concatenate.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_2_path}. Check file format.")
        return

    # Direct concatenation of the two lists
    combined_data = data_file1 + data_file2
    print(f"DEBUG: Total entries after concatenation: {len(combined_data)}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_combined_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save the concatenated data
    try:
        with open(output_combined_path, 'w') as f:
            json.dump(combined_data, f, indent=2) # Use indent=2 for readability
        print(f"Successfully concatenated JSON files and saved to {output_combined_path}")
    except IOError as e:
        print(f"Error saving concatenated JSON to {output_combined_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration for your specific request matching mycom.json example ---
    json_file_1_input = "my_combined_output.json"
    json_file_2_input = "edited_jdata_200.json"
    
    # Output path for the new combined JSON file
    output_directory = "combined_json_output"
    output_filename = "my_combined_output.json" # Name to match your example's behavior
    output_full_path = os.path.join(output_directory, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Run the concatenation
    concatenate_json_files(json_file_1_input, json_file_2_input, output_full_path)