"""
Enhanced CSV to Ultralytics converter with line simplification and polygon generation.
Simple utility functions for use in notebooks.
"""

import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import math


def douglas_peucker(points, epsilon):
    """Simplify a line using the Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    
    start = np.array(points[0])
    end = np.array(points[-1])
    
    max_dist = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        point = np.array(points[i])
        
        if np.allclose(start, end):
            dist = np.linalg.norm(point - start)
        else:
            dist = abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)
        
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    if max_dist > epsilon:
        left_points = douglas_peucker(points[:max_index + 1], epsilon)
        right_points = douglas_peucker(points[max_index:], epsilon)
        return left_points[:-1] + right_points
    else:
        return [points[0], points[-1]]


def create_buffered_polygon(line_points, buffer_pixels=3):
    """Create a buffered polygon around a line."""
    if len(line_points) < 2:
        return line_points
    
    points = np.array(line_points)
    left_side = []
    right_side = []
    
    for i in range(len(points)):
        if i == 0:
            direction = points[i + 1] - points[i]
        elif i == len(points) - 1:
            direction = points[i] - points[i - 1]
        else:
            incoming = points[i] - points[i - 1]
            outgoing = points[i + 1] - points[i]
            direction = incoming + outgoing
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0])
        
        perpendicular = np.array([-direction[1], direction[0]])
        
        left_point = points[i] + perpendicular * buffer_pixels
        right_point = points[i] - perpendicular * buffer_pixels
        
        left_side.append(tuple(left_point))
        right_side.append(tuple(right_point))
    
    return left_side + list(reversed(right_side))


def csv_to_ultralytics_enhanced(data_directory, single_class=True, 
                               epsilon_factor=0.002, buffer_pixels=3):
    """
    Convert CSV files to Ultralytics format with line simplification and polygon creation.
    
    Args:
        data_directory (str): Path to directory containing either:
                             - 'images/' and 'labels/' folders (single dataset)
                             - Multiple subdirectories each with 'images/' and 'labels/' folders (batch mode)
        single_class (bool): If True, convert all categories to class "0" (default: True)
        epsilon_factor (float): Douglas-Peucker tolerance as fraction of image diagonal
        buffer_pixels (int): Buffer distance in pixels around the line
        
    Returns:
        str: Path to output directory with converted files (or summary for batch mode)
    """
    
    data_path = Path(data_directory)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    # Check if this is a single dataset or batch mode
    if images_dir.exists() and labels_dir.exists():
        # Single dataset mode
        return _process_single_dataset(data_path, single_class, epsilon_factor, buffer_pixels)
    else:
        # Batch mode - check for subdirectories with images/labels folders
        subdirectories = [d for d in data_path.iterdir() 
                         if d.is_dir() and (d / "images").exists() and (d / "labels").exists()]
        
        if not subdirectories:
            print(f"Error: No valid datasets found in {data_directory}")
            print("Expected either:")
            print("  1. A directory with 'images/' and 'labels/' folders")
            print("  2. Subdirectories each containing 'images/' and 'labels/' folders")
            return None
        
        # Process all subdirectories in batch
        return _process_batch_datasets(data_path, subdirectories, single_class, epsilon_factor, buffer_pixels)


def _process_single_dataset(data_path, single_class, epsilon_factor, buffer_pixels):
    """Process a single dataset with images/ and labels/ folders."""
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    output_dir = data_path / "ultralytics_labels_enhanced"
    
def _process_single_dataset(data_path, single_class, epsilon_factor, buffer_pixels):
    """Process a single dataset with images/ and labels/ folders."""
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    output_dir = data_path / "ultralytics_labels_enhanced"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)
    
    print(f"Processing dataset: {data_path.name}")
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Settings: epsilon_factor={epsilon_factor}, buffer_pixels={buffer_pixels}")
    
    total_original_points = 0
    total_simplified_points = 0
    
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
                potential_path = images_dir / f"{label_name}{ext}"
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
                        try:
                            x, y = float(row[0]), float(row[1])
                            coordinates.append((x, y))
                        except ValueError:
                            continue
            
            if len(coordinates) < 2:
                print(f"Not enough coordinates in {csv_file}")
                continue
            
            # Calculate epsilon based on image size
            diagonal = math.sqrt(width**2 + height**2)
            epsilon = diagonal * epsilon_factor
            
            # Apply Douglas-Peucker simplification
            simplified_line = douglas_peucker(coordinates, epsilon)
            
            # Create buffered polygon
            polygon_points = create_buffered_polygon(simplified_line, buffer_pixels)
            
            if len(polygon_points) < 3:
                print(f"Not enough polygon points for {csv_file}")
                continue
            
            # Normalize coordinates to [0, 1]
            normalized_coords = []
            for x, y in polygon_points:
                norm_x = max(0, min(1, x / width))
                norm_y = max(0, min(1, y / height))
                normalized_coords.append((norm_x, norm_y))
            
            # Write to ultralytics format
            output_file = output_dir / f"{label_name}.txt"
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_coords])
            
            # Use class index 0 for all classes if single_class is True
            output_class = "0" if single_class else class_index
            
            # Append to file (multiple classes per image)
            with open(output_file, 'a') as f:
                f.write(f"{output_class} {coords_str}\n")
            
            # Track statistics
            total_original_points += len(coordinates)
            total_simplified_points += len(polygon_points)
            
            reduction = (len(coordinates) - len(polygon_points)) / len(coordinates) * 100
            print(f"Processed class {class_index}: {csv_file.name} -> {output_file.name}")
            print(f"  Points: {len(coordinates)} -> {len(polygon_points)} ({reduction:.1f}% reduction)")
    
    overall_reduction = (total_original_points - total_simplified_points) / total_original_points * 100 if total_original_points > 0 else 0
    print(f"\nOverall reduction: {total_original_points} -> {total_simplified_points} points ({overall_reduction:.1f}%)")
    
    return str(output_dir)


def _process_batch_datasets(data_path, subdirectories, single_class, epsilon_factor, buffer_pixels):
    """Process multiple datasets in batch mode."""
    print(f"BATCH MODE: Found {len(subdirectories)} datasets to process:")
    subdirectories.sort(key=lambda x: x.name)
    
    for subdir in subdirectories:
        print(f"  - {subdir.name}")
    
    print(f"\n{'='*60}")
    print("Starting batch processing...")
    print(f"{'='*60}")
    
    results = {}
    total_datasets = len(subdirectories)
    
    for i, subdir in enumerate(subdirectories, 1):
        print(f"\n[{i}/{total_datasets}] Processing: {subdir.name}")
        print(f"Path: {subdir}")
        print("-" * 40)
        
        try:
            output_path = _process_single_dataset(subdir, single_class, epsilon_factor, buffer_pixels)
            results[subdir.name] = f"SUCCESS - Output: {output_path}"
            print(f"✓ Successfully processed {subdir.name}")
        except Exception as e:
            error_msg = f"ERROR - {str(e)}"
            results[subdir.name] = error_msg
            print(f"✗ Error processing {subdir.name}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for result in results.values() if "SUCCESS" in result)
    error_count = total_datasets - success_count
    
    print(f"Total datasets: {total_datasets}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print()
    
    for dataset_name, result in results.items():
        status = "✓" if "SUCCESS" in result else "✗"
        print(f"{status} {dataset_name}: {result}")
    
    return f"Batch processing complete: {success_count}/{total_datasets} datasets processed successfully"


# Example usage:
if __name__ == "__main__":
    # Convert with default settings (recommended):
    # csv_to_ultralytics_enhanced("/path/to/your/data")
    
    # Custom settings:
    # csv_to_ultralytics_enhanced("/path/to/your/data", 
    #                            single_class=True,      # All classes -> class 0
    #                            epsilon_factor=0.001,   # More detailed (smaller = more detail)
    #                            buffer_pixels=5)        # Larger buffer around lines
    pass
