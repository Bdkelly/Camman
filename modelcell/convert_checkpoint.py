import torch
import os

def convert_checkpoint_to_model_weights(checkpoint_path, output_weights_path):
    """
    Loads a training checkpoint and saves only the model's state_dict
    to a new .pth file, making it easier for direct model loading.

    Args:
        checkpoint_path (str): Path to the input checkpoint file (e.g., 'trained_model_final.pth').
        output_weights_path (str): Path to save the new file containing only model weights.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Load the entire checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        
        # Check if 'model_state_dict' key exists
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("Successfully extracted 'model_state_dict' from checkpoint.")
        else:
            # Fallback for older saves that might directly save model.state_dict() without a wrapper
            model_state_dict = checkpoint
            print("Warning: 'model_state_dict' key not found in checkpoint. Assuming checkpoint is raw state_dict.")

        # Save only the model's state_dict
        torch.save(model_state_dict, output_weights_path)
        print(f"Successfully saved model weights to: {output_weights_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your existing training checkpoint file
    input_checkpoint = 'DQN_Agent_now2.pth' # Or 'checkpoint_epoch_X.pth'

    # Desired path for the new file containing only model weights
    output_model_weights = 'DQN_AGENT_final.pth'

    # Run the conversion script
    convert_checkpoint_to_model_weights(input_checkpoint, output_model_weights)