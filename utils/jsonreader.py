import json
import os # Import os for path checking

def data_fix(json_data_path):
    if not os.path.exists(json_data_path):
        print(f"Error: JSON file not found at {json_data_path}")

    with open(json_data_path, 'r') as f:
        data = json.load(f)
    return data


def infbondbox(json_data_path):
    data = data_fix(json_data_path)
    filtered_data = [d for d in data if d[list(d.keys())[0]]]
    return filtered_data

    


def extbondbox(json_data_path):
    data = data_fix(json_data_path)

    annotations_by_frame = {} # This will be the dictionary to return

    # The provided JSON shows a list of dictionaries at the top level,
    # with the actual video annotations nested under 'box' or 'annotations'.
    if isinstance(data, list):
        for item in data:
            # Handle the structure where 'box' is at the top level, as in your video1p.json
            if 'box' in item and isinstance(item['box'], list):
                for box_annotation_set in item['box']:
                    if 'sequence' in box_annotation_set and isinstance(box_annotation_set['sequence'], list):
                        labels_in_set = box_annotation_set.get('labels', [])
                        # Assuming one label per sequence, or taking the first if multiple are listed
                        label_name = labels_in_set[0] if labels_in_set else 'unknown' # Default to 'unknown' if no label

                        for frame_data in box_annotation_set['sequence']:
                            frame_number = frame_data.get("frame")
                            enabled = frame_data.get("enabled", True) # Annotations can be disabled

                            if frame_number is None or not enabled:
                                # Skip frames without a frame number or disabled annotations
                                continue
                            
                            # Extract all relevant info for the bounding box
                            extracted_box_info = {
                                "x": frame_data.get("x"),
                                "y": frame_data.get("y"),
                                "width": frame_data.get("width"),
                                "height": frame_data.get("height"),
                                "rotation": frame_data.get("rotation", 0), # Default rotation to 0 if not present
                                "time": frame_data.get("time"),
                                "label_name": label_name # Include the label name
                            }
                            
                            # Add to the dictionary, creating a list for the frame if it doesn't exist
                            if frame_number not in annotations_by_frame:
                                annotations_by_frame[frame_number] = []
                            annotations_by_frame[frame_number].append(extracted_box_info)
            
            # Handle alternative Label Studio export structure (more common for image projects)
            elif 'annotations' in item and isinstance(item['annotations'], list):
                for annotation in item['annotations']:
                    if 'result' in annotation and isinstance(annotation['result'], list):
                        for result_item in annotation['result']:
                            if 'value' in result_item and 'sequence' in result_item['value'] and isinstance(result_item['value']['sequence'], list):
                                labels_in_set = result_item['value'].get('labels', [])
                                label_name = labels_in_set[0] if labels_in_set else 'unknown'

                                for frame_data in result_item['value']['sequence']:
                                    frame_number = frame_data.get("frame")
                                    enabled = frame_data.get("enabled", True)

                                    if frame_number is None or not enabled:
                                        continue

                                    extracted_box_info = {
                                        "x": frame_data.get("x"),
                                        "y": frame_data.get("y"),
                                        "width": frame_data.get("width"),
                                        "height": frame_data.get("height"),
                                        "rotation": frame_data.get("rotation", 0),
                                        "time": frame_data.get("time"),
                                        "label_name": label_name
                                    }

                                    if frame_number not in annotations_by_frame:
                                        annotations_by_frame[frame_number] = []
                                    annotations_by_frame[frame_number].append(extracted_box_info)
            else:
                print(f"Warning: Top-level item without 'box' or 'annotations' key, or in an unexpected format. Skipping item with keys: {item.keys()}")
    else:
        print(f"Error: JSON data is not a list. Expected a list of video/task objects. Type found: {type(data)}")

    return annotations_by_frame

if __name__ == "__main__":
    # Use the path to your Label Studio exported JSON
    # Assuming video1p.json is in the same directory as jsonreader.py
    json_path = "/Users/Ben/Documents/dever/python/data/outframes/video1/jdata/video1p.json" 
    if os.path.exists(json_path):
        print("Not real:",json_path)
    
    # Get the dictionary of annotations by frame number
    setframes = extbondbox(json_path)
    for k,v in setframes.items():
        print(k)
        print(v)