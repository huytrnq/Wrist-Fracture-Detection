import os

def filter_yolo_labels(input_dir, output_dir, class_ids_to_filter):
    """
    Filters out specific class labels from YOLO format annotation files.

    Parameters:
    - input_dir: Directory containing the original YOLO annotation files.
    - output_dir: Directory to save the filtered annotation files.
    - class_ids_to_filter: List of class IDs to filter out.

    Returns:
    - None
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id in class_ids_to_filter:
                        parts[0] = '0'  # Replace the class ID with 0
                        outfile.write(' '.join(parts) + '\n')

if __name__ == "__main__":
    # Directory containing the original YOLO annotation files
    input_dir = '/Users/huytrq/Downloads/fracture'
    
    # Directory to save the filtered annotation files
    output_dir = '/Users/huytrq/Downloads/fracture1'
    
    # List of class IDs to filter out
    class_ids_to_filter = [3]  # Replace with the class IDs you want to filter out
    
    filter_yolo_labels(input_dir, output_dir, class_ids_to_filter)
