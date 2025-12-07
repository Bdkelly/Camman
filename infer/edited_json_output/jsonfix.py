import json

video = "video1"
file_path = 'converted_json_output_converted_from_video1p_filtered.json'
videoname = f"/{video}/imgs"
vpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes"

# 1. Read the JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Create the video entry as a separate dictionary
video_entry = {
    "videos": {
        f"{video}": {
            "format": "mp4",
            "path": f"{vpath+videoname}"
        }
    }
}

# Check if the video information is already the first entry
# We check if data is not empty and if the first element matches our video_entry structure
if data and isinstance(data[0], dict) and "videos" in data[0] and video in data[0]["videos"]:
    print(f"'{video}' is already the first entry in {file_path}. No changes made.")
else:
    # 2. Insert the video entry at the beginning of the list (index 0)
    # This will place the video information as the first element in the JSON array.
    data.insert(0, video_entry)

    # 3. Write the updated JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2) # indent makes the JSON human-readable

    print(f"Successfully added '{video}' as the first entry to {file_path}")