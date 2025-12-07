import json
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog

def process_images_and_balls(json_file_path, image_directory, output_directory, batch_size=2):
    """
    Processes image frames, displays detected balls, prompts user for correction,
    and allows saving updated data in batches or on user's request to stop.

    Args:
        json_file_path (str): Path to the jdata.json file.
        image_directory (str): Directory where image files are located (should contain pre-drawn images).
        output_directory (str): Directory to save updated JSON files.
        batch_size (int): Number of images to process before offering to save.
    """
    try:
        with open(json_file_path, 'r') as f:
            all_frames_data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Check file format.")
        return

    # Convert the list of single-entry dictionaries to a single dictionary
    all_frames_data = {}
    for frame_entry in all_frames_data_list:
        all_frames_data.update(frame_entry)

    modified_frames_data = {}
    root = tk.Tk()
    root.withdraw()
    processed_count = 0
    total_frames = len(all_frames_data)
    frame_ids = sorted(list(all_frames_data.keys()))
    should_stop_processing = False

    # Get the first and last frame IDs for the filename
    first_frame_id = frame_ids[0].replace(".jpg", "").replace(".png", "")
    last_frame_id = frame_ids[-1].replace(".jpg", "").replace(".png", "")

    for i, frame_id in enumerate(frame_ids):
        if should_stop_processing:
            break

        detections = all_frames_data[frame_id]
        
        # --- Adjusted Image Path Construction for 'inferred_frame_00435.jpg' ---
        image_filename = f"inferred_{frame_id}"
        image_path = os.path.join(image_directory, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Inferred image not found for JSON frame_id '{frame_id}' at {image_path}. Skipping.")
            modified_frames_data[frame_id] = all_frames_data[frame_id]
            processed_count += 1
            continue

        image_window = tk.Toplevel(root)
        image_window.title(f"Frame: {frame_id} - Processed {processed_count+1}/{total_frames}")
        image_window.attributes('-topmost', True)

        try:
            image_pil = Image.open(image_path).convert("RGB")
            img_tk = ImageTk.PhotoImage(image_pil)
            main_display_frame = tk.Frame(image_window)
            main_display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
            main_display_frame.grid_columnconfigure(0, weight=1)
            main_display_frame.grid_columnconfigure(1, weight=0)
            main_display_frame.grid_rowconfigure(0, weight=1)
            image_label = tk.Label(main_display_frame, image=img_tk)
            image_label.grid(row=0, column=0, sticky="nsew")
            button_frame = tk.Frame(main_display_frame)
            button_frame.grid(row=0, column=1, sticky="ns", padx=10)

            ball_labels_for_display = []
            if detections:
                for idx, det in enumerate(detections):
                    ball_labels_for_display.append(f"{idx+1}. {det['Label']} (x:{det['x_min']}, y:{det['y_min']})")
            else:
                ball_labels_for_display.append("No balls detected.")

            if len(detections) > 1:
                instruction_text = "Multiple balls. Press a number key to select the correct ball or 'n' if none are correct."
                def on_multi_select(event):
                    try:
                        selected_index = int(event.char) - 1
                        if 0 <= selected_index < len(detections):
                            modified_frames_data[frame_id] = [detections[selected_index]]
                            image_window.destroy()
                        else:
                            messagebox.showerror("Invalid Input", "Please press a valid number.", parent=image_window)
                    except (ValueError, IndexError):
                        messagebox.showerror("Invalid Input", "Please press a number key corresponding to a ball.", parent=image_window)
                
                for key_num in range(1, len(detections) + 1):
                    image_window.bind(str(key_num), on_multi_select)

            elif len(detections) == 1:
                instruction_text = "One ball detected. Press 'q' for correct, 'w' for incorrect, or 'n' if none are correct."
                def on_single_select(event):
                    if event.char == 'q':
                        modified_frames_data[frame_id] = detections
                        image_window.destroy()
                    elif event.char == 'w':
                        modified_frames_data[frame_id] = []
                        image_window.destroy()
                    else:
                        messagebox.showerror("Invalid Input", "Please press 'q', 'w', or 'n'.", parent=image_window)
                
                image_window.bind('q', on_single_select)
                image_window.bind('w', on_single_select)
            else:
                instruction_text = "No balls detected. Skipping frame."
                modified_frames_data[frame_id] = []
                image_window.after(1, image_window.destroy)

            def on_none_correct(event):
                nonlocal should_stop_processing
                modified_frames_data[frame_id] = []
                image_window.destroy()

            image_window.bind('n', on_none_correct)
            
            instruction_label = tk.Label(button_frame, text=instruction_text, wraplength=200)
            instruction_label.pack(pady=5, fill=tk.X)
            
            listbox_label = tk.Label(button_frame, text="Detected Balls:")
            listbox_label.pack(pady=(10, 0), fill=tk.X)
            listbox = tk.Listbox(button_frame, height=len(ball_labels_for_display), width=30)
            for label in ball_labels_for_display:
                listbox.insert(tk.END, label)
            listbox.pack(pady=5, fill=tk.X)
            
            def on_continue_click():
                modified_frames_data[frame_id] = all_frames_data[frame_id]
                image_window.destroy()
            
            continue_button = tk.Button(button_frame, text="Continue (Keep All)", command=on_continue_click)
            continue_button.pack(pady=5, fill=tk.X)

            def on_exit_click():
                nonlocal should_stop_processing
                should_stop_processing = True
                image_window.destroy()

            exit_button = tk.Button(button_frame, text="Stop and Save", command=on_exit_click)
            exit_button.pack(pady=5, fill=tk.X)

            image_window.focus_force()
            image_window.protocol("WM_DELETE_WINDOW", lambda: on_exit_click())
            root.wait_window(image_window)

            processed_count += 1
            if not should_stop_processing and (processed_count % batch_size == 0 or processed_count == total_frames):
                if messagebox.askyesno("Save Progress", f"Processed {processed_count} images. Save current progress?", parent=root):
                    # Pass first and last frame IDs
                    save_updated_json(modified_frames_data, output_directory, first_frame_id, last_frame_id)

        except Exception as e:
            print(f"An error occurred while processing {frame_id}: {e}")
            messagebox.showerror("Processing Error", f"An error occurred while processing frame {frame_id}: {e}\nProcessing will continue.", parent=root)
            modified_frames_data[frame_id] = all_frames_data[frame_id]
            processed_count += 1
            if 'image_window' in locals() and image_window.winfo_exists():
                image_window.destroy()

    root.destroy()
    if processed_count > 0 and len(modified_frames_data) > 0:
        if messagebox.askyesno("Final Save", "All images processed. Do you want to save the final updated data?", parent=root):
            # Pass first and last frame IDs for the final save
            save_updated_json(modified_frames_data, output_directory, first_frame_id, last_frame_id)
    else:
        print("No changes to save or processing was stopped and saved earlier.")
    print("Image processing complete.")

def save_updated_json(data_to_save, output_dir, first_frame_id, last_frame_id):
    """
    Saves the updated data to a new JSON file.

    Args:
        data_to_save (dict): The dictionary containing updated frame data.
        output_dir (str): The directory to save the JSON file.
        first_frame_id (str): The ID of the first frame in the data to be saved.
        last_frame_id (str): The ID of the last frame in the data to be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_list_format = [{k: v} for k, v in data_to_save.items()]
    
    # Construct the filename using the first and last frame IDs
    output_filename = f"jdata_{first_frame_id}_{last_frame_id}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(output_list_format, f, indent=4)
        print(f"Updated data saved to: {output_filepath}")
    except IOError as e:
        print(f"Error saving updated JSON to {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    bfolder = "/Users/Ben/Documents/dever/python/ptorch/data/inferred_images/"
    vfolder = "video1/"
    jfolder = bfolder + vfolder + "jdata/"
    json_file = jfolder + "jdata.json"
    image_folder = bfolder + vfolder + "imgs/"
    output_folder = jfolder + "/cleaned_json"
    batch_size = 100

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_images_and_balls(json_file, image_folder, output_folder, batch_size)