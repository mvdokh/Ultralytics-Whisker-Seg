"""
Simple function to convert CSV coordinate files to Ultralytics format.
This can be imported and used in Jupyter notebooks.
"""

import os
import csv
from pathlib import Path
from PIL import Image


def csv_to_ultralytics_simple(data_directory, single_class=True):
    """
    Convert CSV files to Ultralytics format for segmentation.
    
    Args:
        data_directory (str): Path to directory containing 'images/' and 'labels/' folders
        single_class (bool): If True, convert all categories to class "0" (default: True)
        
    Expected structure:
        data_directory/
        ├── images/
        │   ├── img0000000.png
        │   ├── img0000001.png
        │   └── ...
        └── labels/
            ├── 0/
            │   ├── 0000000.csv
            │   ├── 0000001.csv
            │   └── ...
            ├── 1/
            │   └── ...
            └── ...
    
    Returns:
        str: Path to output directory with converted files
    """
    
    data_path = Path(data_directory)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    output_dir = data_path / "ultralytics_labels"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)
    
    print(f"Processing {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    for class_dir in class_dirs:
        class_index = class_dir.name
        
        # Process each CSV file in this class
        for csv_file in class_dir.glob("*.csv"):
            label_name = csv_file.stem
            
            # Find corresponding image
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = images_dir / f"img{label_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if not image_path:
                print(f"No image found for {csv_file}")
                continue
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Read CSV coordinates
            coordinates = []
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        x, y = float(row[0]), float(row[1])
                        # Normalize to [0, 1]
                        norm_x = max(0, min(1, x / width))
                        norm_y = max(0, min(1, y / height))
                        coordinates.append((norm_x, norm_y))
            
            # Write to ultralytics format
            output_file = output_dir / f"{label_name}.txt"
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in coordinates])
            
            # Use class index 0 for all classes if single_class is True
            output_class = "0" if single_class else class_index
            
            # Append to file (multiple classes per image)
            with open(output_file, 'a') as f:
                f.write(f"{output_class} {coords_str}\n")
            
            print(f"Processed class {class_index} -> output class {output_class}: {csv_file.name} -> {output_file.name}")
    
    return str(output_dir)


# Example usage:
if __name__ == "__main__":
    # Convert all classes to single class (default behavior):
    # csv_to_ultralytics_simple("/path/to/your/data")
    
    # Keep original class indices:
    # csv_to_ultralytics_simple("/path/to/your/data", single_class=False)
    pass
