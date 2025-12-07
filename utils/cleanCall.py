import os
from PIL import Image
try:
    from utils.jsonreader import infbondbox
except:
    from .jsonreader import infbondbox

def getSize(frame,img_dir):
    filepath = getFirst(frame,img_dir)
    print(frame,img_dir)
    with Image.open(filepath) as img:
        wid,hei = img.size
    return wid,hei

def getFirst(frame,img_dir):
    first_img = f"{img_dir}/{frame}"
    if frame.endswith((".png",".jpg")):
        ftype = ""
    else:    
        if os.path.exists(f"{first_img}.png"):
            ftype = ".png"
        else:
            ftype = ".jpg"
    first_img = first_img + ftype
    return first_img

def bespokeClean(item,imgpath):
    annotations_by_frame = []
    for box_annotation_set in item['box']:
        if 'sequence' in box_annotation_set and isinstance(box_annotation_set['sequence'], list):
            labels_in_set = box_annotation_set.get('labels', [])
            # Assuming one label per sequence, or taking the first if multiple are listed
            label_name = labels_in_set[0] if labels_in_set else 'unknown' # Default to 'unknown' if no lab
            for frame_data in box_annotation_set['sequence']:
                frame_number = frame_data.get("frame")
                enabled = frame_data.get("enabled", True) # Annotations can be disabl
                if frame_number is None or not enabled:
                    # Skip frames without a frame number or disabled annotations
                    continue
                frame_n = f"frame_{frame_number:05d}.jpg"
                # Extract all relevant info for the bounding box
                extracted_box_info = {
                    "frame":frame_n,
                    "x": frame_data.get("x"),
                    "y": frame_data.get("y"),
                    "width": frame_data.get("width"),
                    "height": frame_data.get("height")
                    
                }
                if frame_number not in annotations_by_frame:
                    annotations_by_frame.append(extracted_box_info)
    newlist = []
    w,h = getSize(annotations_by_frame[0]["frame"],imgpath)
    for frame in annotations_by_frame:
        xmin = int(round((frame['x'] / 100.0) * w,0))
        ymin = int(round((frame['y'] / 100.0) * h,0))
        xmax = int(round(((frame['x'] + frame["width"]) / 100.0) * w,0))
        ymax = int(round(((frame['y'] + frame["height"]) / 100.0) * w,0))
        newdict = {
                    "frame":frame["frame"],
                    "Label":"Ball",
                    "x_min":xmin,
                    "y_min":ymin,
                    "x_max":xmax,
                    "y_max":ymax
        }
        newlist.append(newdict)
    return newlist,w,h

def infClean(data,imgpath):
    framelist = []
    for frames in data:
        for k,v in frames.items():
            newdict = {
                    "frame":k,
                    "Label":"Ball",
                    "x_min":v[0]["x_min"],
                    "y_min":v[0]["y_min"],
                    "x_max":v[0]["x_max"],
                    "y_max":v[0]["y_max"]
            }
            framelist.append(newdict)
    return framelist

def cleanCall(jsonpath,imgpath):
    width,height = 0,0
    data = infbondbox(jsonpath)
    if isinstance(data, list):
        for item in data:
            if 'box' in item and isinstance(item['box'], list):
                clean_data,width,height= bespokeClean(item,imgpath)
            else:
                clean_data = infClean(data,imgpath)
    if width and height == 0:
        width,height = getSize(clean_data[0])
    return clean_data,width,height

if __name__ == "__main__": 
    #jsonpath = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/inferred_images/mlsvideo/jdata/cleaned_json/comb/vb.json"
    #imgpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/mlsvideo/imgs"
    jsonpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p5/dips005.json"
    imgpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p5/imgs"
    #jsonpath = "/Users/Ben/Documents/dever/python/ptorch/data/video1p.json"
    #imgpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/video1/imgs"
    clean_data,widht,height = cleanCall(jsonpath,imgpath)
    for i in clean_data:
        print(i)
        

