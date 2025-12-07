import torch

# Assume you have two .pth files: 'model_A.pth' and 'model_B.pth'

# Step 1: Load the state_dict from each file
state_dict_A = torch.load('checkpoint_epoch_130.pth')
state_dict_B = torch.load('trained_model_final copy.pth')

# Step 2: Create a new, empty dictionary to store the merged parameters
merged_state_dict = {}

# Step 3: Iterate through the keys and values of the first state_dict
# and add them to the merged dictionary
for key, value in state_dict_A.items():
    merged_state_dict[key] = value

# Step 4: Iterate through the second state_dict and add its parameters
# You can choose how to handle conflicting keys. Here, it will overwrite model A's keys.
for key, value in state_dict_B.items():
    merged_state_dict[key] = value

# Step 5: (Optional) Save the new, merged state dictionary to a new file
torch.save(merged_state_dict, 'merged_model.pth')

# Step 6: Load the merged state_dict into a new model
# Assuming 'YourNewModel' is the combined model class
new_model = YourNewModel()
new_model.load_state_dict(merged_state_dict)