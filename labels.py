import os

def filter_and_change_yolo_labels(input_folder, output_folder, class_to_filter, new_class_id=0):
    """
    Filter YOLO label files by a specific class and change the class ID to a new value.
    
    Parameters:
    - input_folder: Path to the folder containing YOLO label files.
    - output_folder: Path to the folder where filtered labels will be saved.
    - class_to_filter: The class ID to filter.
    - new_class_id: The new class ID to set in the filtered labels (default is 0).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_filepath = os.path.join(input_folder, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(input_filepath):
            with open(input_filepath, 'r') as file:
                lines = file.readlines()
            
            # Filter lines by the specified class and change the class ID
            filtered_lines = []
            for line in lines:
                parts = line.split()
                if int(float(parts[0])) == class_to_filter:
                    parts[0] = str(new_class_id)
                    parts.insert(1, "0.99")  # Set the confidence score to 0
                    filtered_lines.append(" ".join(parts) + "\n")
            # Write filtered lines to the output file
            output_filepath = os.path.join(output_folder, filename)
            with open(output_filepath, 'w') as file:
                file.writelines(filtered_lines)
            
            # Print the processed file information
            print(f"Processed {input_filepath} -> {output_filepath}")

# Example usage:
input_folder = "./results/predictions/"
output_folder = "./results/filtered_labels/"
class_to_filter = 0  # Change this to the class you want to filter
new_class_id = 0  # Change this to the new class ID you want to set

filter_and_change_yolo_labels(input_folder, output_folder, class_to_filter, new_class_id)
