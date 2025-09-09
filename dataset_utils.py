import os
import glob
import csv
import numpy as np
import math
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union


def remove_img_prefix(root_directory, dry_run=True):
    """
    Remove 'img' prefix from all image files in subdirectories.
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories with images folders
        dry_run (bool): If True, only print what would be renamed without actually renaming files
    
    Returns:
        dict: Summary of the operation including counts and any errors
    """
    # Convert to Path object for easier manipulation
    root_path = Path(root_directory)
    
    # Check if the root directory exists
    if not root_path.exists():
        return {"error": f"Directory {root_directory} does not exist"}
    
    summary = {
        "total_files_found": 0,
        "files_renamed": 0,
        "errors": [],
        "renamed_files": []
    }
    
    # Find all subdirectories
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            # Look for 'images' folder in each subdirectory
            images_folder = subdir / "images"
            
            if images_folder.exists() and images_folder.is_dir():
                print(f"Processing images folder: {images_folder}")
                
                # Find all files starting with 'img' (case insensitive)
                for img_file in images_folder.glob("img*"):
                    if img_file.is_file():
                        summary["total_files_found"] += 1
                        
                        # Create new filename by removing 'img' prefix
                        old_name = img_file.name
                        
                        # Remove 'img' prefix (case insensitive)
                        if old_name.lower().startswith('img'):
                            # Remove the first 3 characters ('img')
                            new_name = old_name[3:]
                            
                            # If the new name is empty or starts with a dot, skip
                            if not new_name or new_name.startswith('.'):
                                summary["errors"].append(f"Skipping {old_name}: would result in invalid filename")
                                continue
                            
                            new_path = img_file.parent / new_name
                            
                            # Check if target file already exists
                            if new_path.exists():
                                summary["errors"].append(f"Skipping {old_name}: target file {new_name} already exists")
                                continue
                            
                            if dry_run:
                                print(f"  Would rename: {old_name} -> {new_name}")
                                summary["renamed_files"].append({
                                    "old_name": old_name,
                                    "new_name": new_name,
                                    "path": str(img_file.parent)
                                })
                            else:
                                try:
                                    img_file.rename(new_path)
                                    print(f"  Renamed: {old_name} -> {new_name}")
                                    summary["files_renamed"] += 1
                                    summary["renamed_files"].append({
                                        "old_name": old_name,
                                        "new_name": new_name,
                                        "path": str(img_file.parent)
                                    })
                                except Exception as e:
                                    error_msg = f"Error renaming {old_name}: {str(e)}"
                                    print(f"  {error_msg}")
                                    summary["errors"].append(error_msg)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total files found with 'img' prefix: {summary['total_files_found']}")
    if dry_run:
        print(f"Files that would be renamed: {len(summary['renamed_files'])}")
        print("This was a dry run. Set dry_run=False to actually rename files.")
    else:
        print(f"Files successfully renamed: {summary['files_renamed']}")
    
    if summary["errors"]:
        print(f"Errors encountered: {len(summary['errors'])}")
        for error in summary["errors"]:
            print(f"  - {error}")
    
    return summary


def list_img_files(root_directory):
    """
    List all files with 'img' prefix in the directory structure.
    
    Args:
        root_directory (str): Path to the root directory
    
    Returns:
        list: List of file paths that start with 'img'
    """
    root_path = Path(root_directory)
    img_files = []
    
    if not root_path.exists():
        print(f"Directory {root_directory} does not exist")
        return img_files
    
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            images_folder = subdir / "images"
            if images_folder.exists() and images_folder.is_dir():
                for img_file in images_folder.glob("img*"):
                    if img_file.is_file():
                        img_files.append(str(img_file))
    
    return img_files


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


def csv_to_ultralytics_conversion(data_directory, 
                                single_class=True, 
                                epsilon_factor=0.002, 
                                buffer_pixels=3,
                                output_suffix="_enhanced"):
    """
    Convert CSV files to Ultralytics format with line simplification and polygon creation.
    
    Args:
        data_directory (str): Path to directory containing 'images/' and 'labels/' folders
        single_class (bool): If True, convert all categories to class "0" (default: True)
        epsilon_factor (float): Douglas-Peucker tolerance as fraction of image diagonal
        buffer_pixels (int): Buffer distance in pixels around the line
        output_suffix (str): Suffix for output directory name
        
    Returns:
        dict: Summary of conversion results
    """
    
    data_path = Path(data_directory)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    output_dir = data_path / f"ultralytics_labels{output_suffix}"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Validation
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Get all class directories
    class_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {labels_dir}")
    
    print(f"Processing {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Settings: single_class={single_class}, epsilon_factor={epsilon_factor}, buffer_pixels={buffer_pixels}")
    
    # Statistics tracking
    stats = {
        'total_original_points': 0,
        'total_simplified_points': 0,
        'processed_images': set(),
        'successful_conversions': 0,
        'failed_conversions': 0,
        'class_counts': {}
    }
    
    for class_dir in class_dirs:
        class_index = class_dir.name
        stats['class_counts'][class_index] = 0
        
        print(f"\nProcessing class {class_index}...")
        
        # Process each CSV file in this class
        for csv_file in class_dir.glob("*.csv"):
            label_name = csv_file.stem
            
            try:
                # Find corresponding image
                image_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    # Try with img prefix first
                    potential_path = images_dir / f"img{label_name}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break
                    # Try without prefix
                    potential_path = images_dir / f"{label_name}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break
                
                if not image_path:
                    print(f"  ⚠ No image found for {csv_file.name}")
                    stats['failed_conversions'] += 1
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
                    print(f"  ⚠ Not enough coordinates in {csv_file.name}")
                    stats['failed_conversions'] += 1
                    continue
                
                # Calculate epsilon based on image size
                diagonal = math.sqrt(width**2 + height**2)
                epsilon = diagonal * epsilon_factor
                
                # Apply Douglas-Peucker simplification
                simplified_line = douglas_peucker(coordinates, epsilon)
                
                # Create buffered polygon
                polygon_points = create_buffered_polygon(simplified_line, buffer_pixels)
                
                if len(polygon_points) < 3:
                    print(f"  ⚠ Not enough polygon points for {csv_file.name}")
                    stats['failed_conversions'] += 1
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
                
                # Update statistics
                stats['total_original_points'] += len(coordinates)
                stats['total_simplified_points'] += len(polygon_points)
                stats['processed_images'].add(label_name)
                stats['successful_conversions'] += 1
                stats['class_counts'][class_index] += 1
                
                reduction = (len(coordinates) - len(polygon_points)) / len(coordinates) * 100
                print(f"  ✓ {csv_file.name} -> {output_file.name} ({len(coordinates)} -> {len(polygon_points)} points, {reduction:.1f}% reduction)")
                
            except Exception as e:
                print(f"  ✗ Error processing {csv_file.name}: {e}")
                stats['failed_conversions'] += 1
    
    # Calculate overall statistics
    if stats['total_original_points'] > 0:
        overall_reduction = (stats['total_original_points'] - stats['total_simplified_points']) / stats['total_original_points'] * 100
    else:
        overall_reduction = 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Processed images: {len(stats['processed_images'])}")
    print(f"Successful conversions: {stats['successful_conversions']}")
    print(f"Failed conversions: {stats['failed_conversions']}")
    print(f"Overall point reduction: {stats['total_original_points']} -> {stats['total_simplified_points']} ({overall_reduction:.1f}%)")
    print(f"Single class mode: {'ON (all classes -> 0)' if single_class else 'OFF (original class indices)'}")
    
    print(f"\nClass distribution:")
    for class_idx, count in stats['class_counts'].items():
        print(f"  Class {class_idx}: {count} objects")
    
    return {
        'output_directory': str(output_dir),
        'statistics': stats,
        'settings': {
            'single_class': single_class,
            'epsilon_factor': epsilon_factor,
            'buffer_pixels': buffer_pixels
        }
    }


def batch_csv_to_ultralytics(base_directory, **conversion_kwargs):
    """
    Process all subdirectories in base directory with CSV to Ultralytics conversion.
    
    Args:
        base_directory (str): Path to directory containing subdirectories with whisker data
        **conversion_kwargs: Arguments to pass to csv_to_ultralytics_conversion
        
    Returns:
        dict: Results from all conversions
    """
    
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist!")
        return {}
    
    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {base_directory}")
        return {}
    
    print(f"Found {len(subdirs)} directories to process:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")
    
    print(f"\nBatch processing settings:")
    for key, value in conversion_kwargs.items():
        print(f"  {key}: {value}")
    print()
    
    # Track results
    results = {}
    successful = []
    failed = []
    
    # Process each subdirectory
    for i, subdir in enumerate(subdirs, 1):
        print(f"[{i}/{len(subdirs)}] Processing {subdir.name}...")
        
        try:
            # Check if directory has the expected structure
            images_dir = subdir / "images"
            labels_dir = subdir / "labels"
            
            if not images_dir.exists():
                print(f"  ⚠ Warning: No 'images' folder found in {subdir.name}")
                continue
                
            if not labels_dir.exists():
                print(f"  ⚠ Warning: No 'labels' folder found in {subdir.name}")
                continue
            
            # Run the conversion
            result = csv_to_ultralytics_conversion(str(subdir), **conversion_kwargs)
            results[subdir.name] = result
            successful.append(subdir.name)
            
        except Exception as e:
            failed.append((subdir.name, str(e)))
            print(f"  ✗ Failed to process {subdir.name}: {e}")
        
        print()  # Add spacing between directories
    
    # Print batch summary
    print("="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total directories: {len(subdirs)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful:")
        for name in successful:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")
    
    return results


def douglas_peucker(points, epsilon):
    """
    Simplify a line using Douglas-Peucker algorithm.
    
    Args:
        points: List of (x, y) tuples
        epsilon: Tolerance for simplification
        
    Returns:
        List of simplified (x, y) tuples
    """
    if len(points) <= 2:
        return points
    
    # Find the point with maximum distance from line between first and last points
    dmax = 0
    index = 0
    end = len(points) - 1
    
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
    
    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index+1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)
        
        # Build the result list
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end]]
    
    return result


def perpendicular_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line."""
    if line_start[0] == line_end[0] and line_start[1] == line_end[1]:
        # Line is actually a point
        return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
    
    A = line_end[0] - line_start[0]
    B = line_end[1] - line_start[1] 
    C = line_start[0] * line_end[1] - line_end[0] * line_start[1]
    
    distance = abs(A * point[1] - B * point[0] + C) / math.sqrt(A**2 + B**2)
    return distance


def create_buffered_polygon(points, buffer_pixels=2):
    """
    Create a buffered polygon from line points.
    
    Args:
        points: List of (x, y) coordinate tuples
        buffer_pixels: Buffer distance in pixels
        
    Returns:
        List of (x, y) tuples representing polygon boundary
    """
    if len(points) < 2:
        return points
    
    # Create LineString from points
    line = LineString(points)
    
    # Create buffer around the line
    buffered = line.buffer(buffer_pixels, cap_style=2, join_style=2)  # Square caps and joins
    
    # Extract exterior coordinates
    if hasattr(buffered, 'exterior'):
        coords = list(buffered.exterior.coords)
    else:
        # If buffered shape is more complex, get the convex hull
        coords = list(buffered.convex_hull.exterior.coords)
    
    # Remove duplicate last point (shapely includes it for closure)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    
    return coords


def csv_to_ultralytics_conversion(input_directory, 
                                single_class=False, 
                                epsilon_factor=0.002,
                                buffer_pixels=2,
                                output_suffix="_enhanced"):
    """
    Convert CSV whisker labels to Ultralytics segmentation format with enhanced processing.
    
    Args:
        input_directory: Directory containing 'images' and 'labels' folders
        single_class: If True, convert all labels to class 0
        epsilon_factor: Tolerance factor for Douglas-Peucker simplification (relative to image size)
        buffer_pixels: Buffer distance around lines to create polygons
        output_suffix: Suffix for output directory name
        
    Returns:
        Dict with conversion statistics and results
    """
    input_path = Path(input_directory)
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    
    # Create output directory
    output_dir = input_path / f"ultralytics_labels{output_suffix}"
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
        image_files.extend(images_dir.glob(ext.upper()))
    
    print(f"Found {len(image_files)} images to process")
    
    statistics = {
        'processed_images': [],
        'successful_conversions': 0,
        'total_original_points': 0,
        'total_simplified_points': 0,
        'class_counts': {}
    }
    
    for image_file in image_files:
        try:
            # Get image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size
            
            # Calculate epsilon based on image size
            epsilon = min(img_width, img_height) * epsilon_factor
            
            image_stem = image_file.stem
            statistics['processed_images'].append(image_stem)
            
            # Look for corresponding CSV file
            csv_file = labels_dir / f"{image_stem}.csv"
            if not csv_file.exists():
                print(f"No CSV file found for {image_stem}")
                continue
            
            # Create output file
            output_file = output_dir / f"{image_stem}.txt"
            
            with open(csv_file, 'r') as infile, open(output_file, 'w') as outfile:
                csv_reader = csv.reader(infile)
                
                for row in csv_reader:
                    if len(row) < 3:  # Need at least class and 2 coordinates
                        continue
                    
                    try:
                        # Parse class (first column)
                        original_class = int(row[0])
                        
                        # Count original classes
                        if original_class not in statistics['class_counts']:
                            statistics['class_counts'][original_class] = 0
                        statistics['class_counts'][original_class] += 1
                        
                        # Convert to single class if requested
                        class_id = 0 if single_class else original_class
                        
                        # Parse coordinates (remaining columns, should be even number)
                        coords = [float(x) for x in row[1:]]
                        if len(coords) % 2 != 0:
                            print(f"Warning: Odd number of coordinates in {image_stem}, row: {row}")
                            continue
                        
                        # Group into (x, y) pairs
                        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        statistics['total_original_points'] += len(points)
                        
                        # Apply Douglas-Peucker simplification
                        simplified_points = douglas_peucker(points, epsilon)
                        
                        # Create buffered polygon
                        polygon_points = create_buffered_polygon(simplified_points, buffer_pixels)
                        statistics['total_simplified_points'] += len(polygon_points)
                        
                        # Normalize coordinates to [0, 1] range
                        normalized_coords = []
                        for x, y in polygon_points:
                            norm_x = max(0, min(1, x / img_width))
                            norm_y = max(0, min(1, y / img_height))
                            normalized_coords.extend([norm_x, norm_y])
                        
                        # Write to output file
                        coord_str = ' '.join(f"{coord:.6f}" for coord in normalized_coords)
                        outfile.write(f"{class_id} {coord_str}\n")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error processing row in {image_stem}: {row}, Error: {e}")
                        continue
            
            statistics['successful_conversions'] += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"\nConversion completed!")
    print(f"Processed: {len(statistics['processed_images'])} images")
    print(f"Successful: {statistics['successful_conversions']} conversions")
    print(f"Point reduction: {statistics['total_original_points']} -> {statistics['total_simplified_points']}")
    print(f"Output directory: {output_dir}")
    
    if single_class:
        print(f"Single class mode: All labels converted to class 0")
        print(f"Original classes found: {list(statistics['class_counts'].keys())}")
    
    return {
        'output_directory': str(output_dir),
        'statistics': statistics
    }


def batch_csv_to_ultralytics(root_directory, **conversion_kwargs):
    """
    Run CSV to Ultralytics conversion on all subdirectories.
    
    Args:
        root_directory: Root directory containing subdirectories with images/labels
        **conversion_kwargs: Arguments to pass to csv_to_ultralytics_conversion
        
    Returns:
        Dict with results for each processed directory
    """
    root_path = Path(root_directory)
    
    if not root_path.exists():
        raise ValueError(f"Root directory does not exist: {root_directory}")
    
    # Find all subdirectories that contain both 'images' and 'labels' folders
    subdirs = []
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            images_dir = subdir / "images"
            labels_dir = subdir / "labels"
            if images_dir.exists() and labels_dir.exists():
                subdirs.append(subdir)
    
    print(f"Found {len(subdirs)} directories with images and labels folders")
    
    results = {}
    successful = []
    failed = []
    
    for subdir in subdirs:
        print(f"Processing: {subdir.name}")
        print("-" * 40)
        
        try:
            images_dir = subdir / "images"
            labels_dir = subdir / "labels"
            
            if not images_dir.exists():
                print(f"  ⚠ Warning: No 'images' folder found in {subdir.name}")
                continue
                
            if not labels_dir.exists():
                print(f"  ⚠ Warning: No 'labels' folder found in {subdir.name}")
                continue
            
            # Run the conversion
            result = csv_to_ultralytics_conversion(str(subdir), **conversion_kwargs)
            results[subdir.name] = result
            successful.append(subdir.name)
            
        except Exception as e:
            failed.append((subdir.name, str(e)))
            print(f"  ✗ Failed to process {subdir.name}: {e}")
        
        print()  # Add spacing between directories
    
    # Print batch summary
    print("="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total directories: {len(subdirs)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful:")
        for name in successful:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")
    
    return results
