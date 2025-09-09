#!/usr/bin/env python3
"""
Convert CSV files with x,y coordinates to Ultralytics segmentation format.

Expected directory structure:
.
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
    │   ├── 0000000.csv
    │   └── ...
    └── ...

CSV format: Each file contains x,y coordinates (comma-delimited)
- First column: x values
- Second column: y values
- Each row: one pixel
- Ordered from base to tip

Output: Ultralytics format text files
Format: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
Coordinates are normalized to [0, 1] range.
"""

import os
import csv
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def read_csv_coordinates(csv_path):
    """
    Read x,y coordinates from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        list: List of (x, y) tuples
    """
    coordinates = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        coordinates.append((x, y))
                    except ValueError:
                        print(f"Warning: Skipping invalid row in {csv_path}: {row}")
                        continue
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return []
    
    return coordinates


def normalize_coordinates(coordinates, image_width, image_height):
    """
    Normalize coordinates to [0, 1] range.
    
    Args:
        coordinates (list): List of (x, y) tuples
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        
    Returns:
        list: List of normalized (x, y) tuples
    """
    normalized = []
    for x, y in coordinates:
        norm_x = x / image_width
        norm_y = y / image_height
        # Clamp to [0, 1] range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        normalized.append((norm_x, norm_y))
    
    return normalized


def get_image_dimensions(image_path):
    """
    Get image dimensions.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        tuple: (width, height) or None if error
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # PIL returns (width, height)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def find_corresponding_image(image_dir, label_filename):
    """
    Find the corresponding image file for a label file.
    
    Args:
        image_dir (Path): Directory containing images
        label_filename (str): Name of the label file (without extension)
        
    Returns:
        Path or None: Path to corresponding image file
    """
    # Common image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Try to find image with same base name
    for ext in image_extensions:
        # Try with 'img' prefix
        img_path = image_dir / f"img{label_filename}{ext}"
        if img_path.exists():
            return img_path
        
        # Try without prefix
        img_path = image_dir / f"{label_filename}{ext}"
        if img_path.exists():
            return img_path
    
    return None


def convert_csv_to_ultralytics(data_dir, output_dir=None, single_class=True):
    """
    Convert CSV files to Ultralytics format.
    
    Args:
        data_dir (str): Root directory containing images/ and labels/ folders
        output_dir (str): Output directory (default: data_dir/ultralytics_labels)
        single_class (bool): If True, convert all categories to class "0" (default: True)
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if output_dir is None:
        output_path = data_path / "ultralytics_labels"
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if required directories exist
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    # Get all class directories (0, 1, 2, etc.)
    class_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"No class directories found in {labels_dir}")
        return
    
    # Sort class directories by name
    class_dirs.sort(key=lambda x: x.name)
    
    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    
    # Process each image
    processed_images = set()
    
    for class_dir in class_dirs:
        class_index = class_dir.name
        print(f"\nProcessing class {class_index}...")
        
        # Get all CSV files in this class directory
        csv_files = list(class_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            label_filename = csv_file.stem  # filename without extension
            
            # Find corresponding image
            image_path = find_corresponding_image(images_dir, label_filename)
            if image_path is None:
                print(f"Warning: No corresponding image found for {csv_file}")
                continue
            
            # Get image dimensions
            dimensions = get_image_dimensions(image_path)
            if dimensions is None:
                print(f"Warning: Could not read image dimensions for {image_path}")
                continue
            
            image_width, image_height = dimensions
            
            # Read coordinates from CSV
            coordinates = read_csv_coordinates(csv_file)
            if not coordinates:
                print(f"Warning: No valid coordinates found in {csv_file}")
                continue
            
            # Normalize coordinates
            normalized_coords = normalize_coordinates(coordinates, image_width, image_height)
            
            # Create output filename
            output_filename = f"{label_filename}.txt"
            output_file_path = output_path / output_filename
            
            # Create the ultralytics format line for this class
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_coords])
            
            # Use class index 0 for all classes if single_class is True
            output_class = "0" if single_class else class_index
            line = f"{output_class} {coords_str}"
            
            # Append to output file (multiple classes can contribute to same image)
            with open(output_file_path, 'a') as f:
                f.write(line + "\n")
            
            processed_images.add(label_filename)
            print(f"  Processed {csv_file.name} -> {output_filename} ({len(coordinates)} points)")
    
    print(f"\nConversion complete!")
    print(f"Processed {len(processed_images)} unique images")
    print(f"Output files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV coordinate files to Ultralytics segmentation format")
    parser.add_argument("data_dir", help="Root directory containing images/ and labels/ folders")
    parser.add_argument("-o", "--output", help="Output directory (default: data_dir/ultralytics_labels)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--multi-class", action="store_true", 
                       help="Keep original class indices (default: convert all to class 0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    # single_class is True by default, False if --multi-class is specified
    single_class = not args.multi_class
    
    try:
        convert_csv_to_ultralytics(args.data_dir, args.output, single_class)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
