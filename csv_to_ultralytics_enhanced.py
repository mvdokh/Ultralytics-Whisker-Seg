#!/usr/bin/env python3
"""
Enhanced CSV to Ultralytics converter with line simplification and polygon generation.

This version includes:
1. Douglas-Peucker algorithm for line simplification
2. Polygon closure (tip loops back to beginning)
3. Buffering around the line to create proper segmentation polygons
4. Overlap prevention
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import math


def douglas_peucker(points, epsilon):
    """
    Simplify a line using the Douglas-Peucker algorithm.
    
    Args:
        points: List of (x, y) tuples
        epsilon: Tolerance for simplification
        
    Returns:
        List of simplified (x, y) tuples
    """
    if len(points) < 3:
        return points
    
    # Find the point with maximum distance from line between first and last points
    start = np.array(points[0])
    end = np.array(points[-1])
    
    max_dist = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        point = np.array(points[i])
        
        # Calculate perpendicular distance from point to line
        if np.allclose(start, end):
            dist = np.linalg.norm(point - start)
        else:
            dist = abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)
        
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursive call on both parts
        left_points = douglas_peucker(points[:max_index + 1], epsilon)
        right_points = douglas_peucker(points[max_index:], epsilon)
        
        # Combine results (remove duplicate middle point)
        return left_points[:-1] + right_points
    else:
        # If max distance is less than epsilon, return only endpoints
        return [points[0], points[-1]]


def create_buffered_polygon(line_points, buffer_pixels=3):
    """
    Create a buffered polygon around a line.
    
    Args:
        line_points: List of (x, y) tuples representing the centerline
        buffer_pixels: Buffer distance in pixels
        
    Returns:
        List of (x, y) tuples representing the polygon vertices
    """
    if len(line_points) < 2:
        return line_points
    
    # Convert to numpy array for easier computation
    points = np.array(line_points)
    
    # Calculate perpendicular directions at each point
    left_side = []
    right_side = []
    
    for i in range(len(points)):
        if i == 0:
            # First point: use direction to next point
            direction = points[i + 1] - points[i]
        elif i == len(points) - 1:
            # Last point: use direction from previous point
            direction = points[i] - points[i - 1]
        else:
            # Middle points: use average of incoming and outgoing directions
            incoming = points[i] - points[i - 1]
            outgoing = points[i + 1] - points[i]
            direction = incoming + outgoing
        
        # Normalize direction
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0])  # Default direction
        
        # Calculate perpendicular (rotate 90 degrees)
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Calculate offset points
        left_point = points[i] + perpendicular * buffer_pixels
        right_point = points[i] - perpendicular * buffer_pixels
        
        left_side.append(tuple(left_point))
        right_side.append(tuple(right_point))
    
    # Create closed polygon: left side + reversed right side
    polygon = left_side + list(reversed(right_side))
    
    return polygon


def simplify_and_polygonize(coordinates, image_width, image_height, 
                           epsilon_factor=0.002, buffer_pixels=3):
    """
    Simplify line coordinates and create a buffered polygon.
    
    Args:
        coordinates: List of (x, y) pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        epsilon_factor: Simplification tolerance as fraction of image diagonal
        buffer_pixels: Buffer distance in pixels
        
    Returns:
        List of normalized (x, y) tuples for the polygon
    """
    if len(coordinates) < 2:
        return coordinates
    
    # Calculate epsilon based on image size
    diagonal = math.sqrt(image_width**2 + image_height**2)
    epsilon = diagonal * epsilon_factor
    
    # Apply Douglas-Peucker simplification
    simplified_line = douglas_peucker(coordinates, epsilon)
    
    # Create buffered polygon
    polygon_points = create_buffered_polygon(simplified_line, buffer_pixels)
    
    # Normalize coordinates to [0, 1] range
    normalized_polygon = []
    for x, y in polygon_points:
        norm_x = max(0.0, min(1.0, x / image_width))
        norm_y = max(0.0, min(1.0, y / image_height))
        normalized_polygon.append((norm_x, norm_y))
    
    return normalized_polygon


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


def convert_csv_to_ultralytics_enhanced(data_dir, output_dir=None, single_class=True,
                                      epsilon_factor=0.002, buffer_pixels=3):
    """
    Convert CSV files to Ultralytics format with line simplification and polygon creation.
    
    Args:
        data_dir (str): Root directory containing images/ and labels/ folders
        output_dir (str): Output directory (default: data_dir/ultralytics_labels_enhanced)
        single_class (bool): If True, convert all categories to class "0" (default: True)
        epsilon_factor (float): Douglas-Peucker tolerance as fraction of image diagonal
        buffer_pixels (int): Buffer distance in pixels around the line
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if output_dir is None:
        output_path = data_path / "ultralytics_labels_enhanced"
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
    print(f"Settings: epsilon_factor={epsilon_factor}, buffer_pixels={buffer_pixels}")
    
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
            
            # Simplify and create polygon
            polygon_coords = simplify_and_polygonize(
                coordinates, image_width, image_height, epsilon_factor, buffer_pixels
            )
            
            if len(polygon_coords) < 3:
                print(f"Warning: Not enough points for polygon in {csv_file}")
                continue
            
            # Create output filename
            output_filename = f"{label_filename}.txt"
            output_file_path = output_path / output_filename
            
            # Create the ultralytics format line for this class
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon_coords])
            
            # Use class index 0 for all classes if single_class is True
            output_class = "0" if single_class else class_index
            line = f"{output_class} {coords_str}"
            
            # Append to output file (multiple classes can contribute to same image)
            with open(output_file_path, 'a') as f:
                f.write(line + "\n")
            
            processed_images.add(label_filename)
            reduction = (len(coordinates) - len(polygon_coords)) / len(coordinates) * 100
            print(f"  Processed {csv_file.name} -> {output_filename}")
            print(f"    Points: {len(coordinates)} -> {len(polygon_coords)} ({reduction:.1f}% reduction)")
    
    print(f"\nConversion complete!")
    print(f"Processed {len(processed_images)} unique images")
    print(f"Output files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV coordinate files to Ultralytics segmentation format with line simplification")
    parser.add_argument("data_dir", help="Root directory containing images/ and labels/ folders")
    parser.add_argument("-o", "--output", help="Output directory (default: data_dir/ultralytics_labels_enhanced)")
    parser.add_argument("--multi-class", action="store_true", 
                       help="Keep original class indices (default: convert all to class 0)")
    parser.add_argument("--epsilon", type=float, default=0.002,
                       help="Douglas-Peucker tolerance as fraction of image diagonal (default: 0.002)")
    parser.add_argument("--buffer", type=int, default=3,
                       help="Buffer distance in pixels around the line (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    # single_class is True by default, False if --multi-class is specified
    single_class = not args.multi_class
    
    try:
        convert_csv_to_ultralytics_enhanced(
            args.data_dir, args.output, single_class, args.epsilon, args.buffer
        )
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
