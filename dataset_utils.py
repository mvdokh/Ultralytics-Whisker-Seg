import os
import glob
from pathlib import Path


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


def reorganize_label_folders(root_directory, dry_run=True):
    """
    Reorganize label folders by deleting 'labels' and renaming 'ultralytics_labels_enhanced' to 'labels'.
    
    This function processes all subdirectories in the root directory and:
    1. Deletes the existing 'labels' folder
    2. Renames 'ultralytics_labels_enhanced' folder to 'labels'
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories with label folders
        dry_run (bool): If True, only print what would be done without actually modifying files
    
    Returns:
        dict: Summary of the operation including counts and any errors
    """
    import shutil
    
    # Convert to Path object for easier manipulation
    root_path = Path(root_directory)
    
    # Check if the root directory exists
    if not root_path.exists():
        return {"error": f"Directory {root_directory} does not exist"}
    
    summary = {
        "total_subdirs_found": 0,
        "subdirs_processed": 0,
        "labels_folders_deleted": 0,
        "ultralytics_folders_renamed": 0,
        "errors": [],
        "processed_dirs": []
    }
    
    print(f"{'='*60}")
    print(f"FOLDER REORGANIZATION: {root_directory}")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL OPERATION'}")
    print()
    
    # Find all subdirectories
    subdirectories = [d for d in root_path.iterdir() if d.is_dir()]
    subdirectories.sort(key=lambda x: x.name)
    
    summary["total_subdirs_found"] = len(subdirectories)
    
    for subdir in subdirectories:
        subdir_name = subdir.name
        labels_folder = subdir / "labels"
        ultralytics_folder = subdir / "ultralytics_labels_enhanced"
        
        print(f"Processing: {subdir_name}")
        
        # Check what folders exist
        has_labels = labels_folder.exists()
        has_ultralytics = ultralytics_folder.exists()
        
        print(f"  labels folder exists: {has_labels}")
        print(f"  ultralytics_labels_enhanced folder exists: {has_ultralytics}")
        
        if not has_ultralytics:
            error_msg = f"No ultralytics_labels_enhanced folder found in {subdir_name}"
            print(f"  WARNING: {error_msg}")
            summary["errors"].append(error_msg)
            continue
        
        # Process this directory
        dir_summary = {
            "name": subdir_name,
            "labels_deleted": False,
            "ultralytics_renamed": False,
            "errors": []
        }
        
        try:
            # Step 1: Delete labels folder if it exists
            if has_labels:
                if dry_run:
                    print(f"  Would DELETE: {labels_folder}")
                else:
                    shutil.rmtree(labels_folder)
                    print(f"  DELETED: {labels_folder}")
                    summary["labels_folders_deleted"] += 1
                dir_summary["labels_deleted"] = True
            else:
                print(f"  No labels folder to delete")
            
            # Step 2: Rename ultralytics_labels_enhanced to labels
            new_labels_path = subdir / "labels"
            if dry_run:
                print(f"  Would RENAME: {ultralytics_folder} -> {new_labels_path}")
            else:
                ultralytics_folder.rename(new_labels_path)
                print(f"  RENAMED: ultralytics_labels_enhanced -> labels")
                summary["ultralytics_folders_renamed"] += 1
            dir_summary["ultralytics_renamed"] = True
            
            summary["subdirs_processed"] += 1
            print(f"  ✓ Successfully processed {subdir_name}")
            
        except Exception as e:
            error_msg = f"Error processing {subdir_name}: {str(e)}"
            print(f"  ✗ {error_msg}")
            summary["errors"].append(error_msg)
            dir_summary["errors"].append(str(e))
        
        summary["processed_dirs"].append(dir_summary)
        print()
    
    # Print final summary
    print(f"{'='*60}")
    print("REORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total subdirectories found: {summary['total_subdirs_found']}")
    print(f"Subdirectories processed: {summary['subdirs_processed']}")
    
    if dry_run:
        labels_would_delete = sum(1 for d in summary['processed_dirs'] if d['labels_deleted'])
        ultralytics_would_rename = sum(1 for d in summary['processed_dirs'] if d['ultralytics_renamed'])
        print(f"Labels folders that would be deleted: {labels_would_delete}")
        print(f"Ultralytics folders that would be renamed: {ultralytics_would_rename}")
        print("\nThis was a DRY RUN. Set dry_run=False to actually reorganize folders.")
    else:
        print(f"Labels folders deleted: {summary['labels_folders_deleted']}")
        print(f"Ultralytics folders renamed to labels: {summary['ultralytics_folders_renamed']}")
    
    if summary["errors"]:
        print(f"\nErrors encountered: {len(summary['errors'])}")
        for error in summary["errors"]:
            print(f"  - {error}")
    
    print()
    
    return summary


def split_train_val(root_directory, train_ratio=0.8, dry_run=True, random_seed=42):
    """
    Split images and labels into train/val sets for all subdirectories.
    
    This function processes all subdirectories in the root directory and:
    1. Creates 'train' and 'val' folders inside each 'images' and 'labels' folder
    2. Splits images with the specified train/val ratio
    3. Ensures corresponding labels go to the same split as their images
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories
        train_ratio (float): Ratio of data to use for training (default: 0.8 = 80%)
        dry_run (bool): If True, only print what would be done without moving files
        random_seed (int): Random seed for reproducible splits
    
    Returns:
        dict: Summary of the operation including counts and any errors
    """
    import shutil
    import random
    from collections import defaultdict
    
    # Set random seed for reproducible results
    random.seed(random_seed)
    
    # Convert to Path object for easier manipulation
    root_path = Path(root_directory)
    
    # Check if the root directory exists
    if not root_path.exists():
        return {"error": f"Directory {root_directory} does not exist"}
    
    summary = {
        "total_subdirs_found": 0,
        "subdirs_processed": 0,
        "total_images_processed": 0,
        "total_labels_processed": 0,
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "errors": [],
        "processed_dirs": []
    }
    
    print(f"{'='*70}")
    print(f"TRAIN/VAL SPLIT: {root_directory}")
    print(f"{'='*70}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL OPERATION'}")
    print(f"Train ratio: {train_ratio:.1%} | Val ratio: {1-train_ratio:.1%}")
    print(f"Random seed: {random_seed}")
    print()
    
    # Find all subdirectories
    subdirectories = [d for d in root_path.iterdir() if d.is_dir()]
    subdirectories.sort(key=lambda x: x.name)
    
    summary["total_subdirs_found"] = len(subdirectories)
    
    for subdir in subdirectories:
        subdir_name = subdir.name
        images_folder = subdir / "images"
        labels_folder = subdir / "labels"
        
        print(f"Processing: {subdir_name}")
        
        # Check if required folders exist
        if not images_folder.exists():
            error_msg = f"No images folder found in {subdir_name}"
            print(f"  WARNING: {error_msg}")
            summary["errors"].append(error_msg)
            continue
            
        if not labels_folder.exists():
            error_msg = f"No labels folder found in {subdir_name}"
            print(f"  WARNING: {error_msg}")
            summary["errors"].append(error_msg)
            continue
        
        # Process this directory
        dir_summary = {
            "name": subdir_name,
            "images_found": 0,
            "labels_found": 0,
            "train_images": 0,
            "val_images": 0,
            "train_labels": 0,
            "val_labels": 0,
            "errors": []
        }
        
        try:
            # Get all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(images_folder.glob(f"*{ext}"))
                image_files.extend(images_folder.glob(f"*{ext.upper()}"))
            
            # Get basenames (without extensions) for matching
            image_basenames = [f.stem for f in image_files]
            dir_summary["images_found"] = len(image_files)
            
            print(f"  Found {len(image_files)} images")
            
            if len(image_files) == 0:
                print(f"  No images found, skipping...")
                continue
            
            # Get all label files that match image basenames
            label_files = []
            label_basenames = []
            for img_basename in image_basenames:
                label_file = labels_folder / f"{img_basename}.txt"
                if label_file.exists():
                    label_files.append(label_file)
                    label_basenames.append(img_basename)
            
            dir_summary["labels_found"] = len(label_files)
            print(f"  Found {len(label_files)} corresponding labels")
            
            # Create train/val directories
            train_images_dir = images_folder / "train"
            val_images_dir = images_folder / "val"
            train_labels_dir = labels_folder / "train"
            val_labels_dir = labels_folder / "val"
            
            # Create directories if not in dry run mode
            if not dry_run:
                train_images_dir.mkdir(exist_ok=True)
                val_images_dir.mkdir(exist_ok=True)
                train_labels_dir.mkdir(exist_ok=True)
                val_labels_dir.mkdir(exist_ok=True)
            else:
                print(f"  Would create directories: train/ and val/ in images/ and labels/")
            
            # Shuffle the basenames for random split
            paired_basenames = list(set(image_basenames) & set(label_basenames))
            random.shuffle(paired_basenames)
            
            # Calculate split point
            n_total = len(paired_basenames)
            n_train = int(n_total * train_ratio)
            
            train_basenames = paired_basenames[:n_train]
            val_basenames = paired_basenames[n_train:]
            
            print(f"  Split: {len(train_basenames)} train, {len(val_basenames)} val")
            
            # Move/copy files to train set
            for basename in train_basenames:
                # Find the image file with this basename
                image_file = None
                for img_file in image_files:
                    if img_file.stem == basename:
                        image_file = img_file
                        break
                
                if image_file:
                    train_img_path = train_images_dir / image_file.name
                    if dry_run:
                        print(f"    Would move: {image_file.name} -> images/train/")
                    else:
                        shutil.move(str(image_file), str(train_img_path))
                    dir_summary["train_images"] += 1
                
                # Move corresponding label file
                label_file = labels_folder / f"{basename}.txt"
                if label_file.exists():
                    train_label_path = train_labels_dir / f"{basename}.txt"
                    if dry_run:
                        print(f"    Would move: {basename}.txt -> labels/train/")
                    else:
                        shutil.move(str(label_file), str(train_label_path))
                    dir_summary["train_labels"] += 1
            
            # Move/copy files to val set
            for basename in val_basenames:
                # Find the image file with this basename
                image_file = None
                for img_file in image_files:
                    if img_file.stem == basename:
                        image_file = img_file
                        break
                
                if image_file:
                    val_img_path = val_images_dir / image_file.name
                    if dry_run:
                        print(f"    Would move: {image_file.name} -> images/val/")
                    else:
                        shutil.move(str(image_file), str(val_img_path))
                    dir_summary["val_images"] += 1
                
                # Move corresponding label file
                label_file = labels_folder / f"{basename}.txt"
                if label_file.exists():
                    val_label_path = val_labels_dir / f"{basename}.txt"
                    if dry_run:
                        print(f"    Would move: {basename}.txt -> labels/val/")
                    else:
                        shutil.move(str(label_file), str(val_label_path))
                    dir_summary["val_labels"] += 1
            
            summary["subdirs_processed"] += 1
            summary["total_images_processed"] += dir_summary["train_images"] + dir_summary["val_images"]
            summary["total_labels_processed"] += dir_summary["train_labels"] + dir_summary["val_labels"]
            summary["train_images"] += dir_summary["train_images"]
            summary["val_images"] += dir_summary["val_images"]
            summary["train_labels"] += dir_summary["train_labels"]
            summary["val_labels"] += dir_summary["val_labels"]
            
            print(f"  ✓ Successfully processed {subdir_name}")
            
        except Exception as e:
            error_msg = f"Error processing {subdir_name}: {str(e)}"
            print(f"  ✗ {error_msg}")
            summary["errors"].append(error_msg)
            dir_summary["errors"].append(str(e))
        
        summary["processed_dirs"].append(dir_summary)
        print()
    
    # Print final summary
    print(f"{'='*70}")
    print("TRAIN/VAL SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total subdirectories found: {summary['total_subdirs_found']}")
    print(f"Subdirectories processed: {summary['subdirs_processed']}")
    print(f"Total images processed: {summary['total_images_processed']}")
    print(f"Total labels processed: {summary['total_labels_processed']}")
    print()
    
    if dry_run:
        print("FILES THAT WOULD BE MOVED:")
        print(f"  Train images: {summary['train_images']}")
        print(f"  Val images: {summary['val_images']}")
        print(f"  Train labels: {summary['train_labels']}")
        print(f"  Val labels: {summary['val_labels']}")
        print(f"\nTrain/Val ratio: {summary['train_images']/(summary['train_images']+summary['val_images']):.1%} / {summary['val_images']/(summary['train_images']+summary['val_images']):.1%}")
        print("\nThis was a DRY RUN. Set dry_run=False to actually move files.")
    else:
        print("FILES MOVED:")
        print(f"  Train images: {summary['train_images']}")
        print(f"  Val images: {summary['val_images']}")
        print(f"  Train labels: {summary['train_labels']}")
        print(f"  Val labels: {summary['val_labels']}")
        if summary['train_images'] + summary['val_images'] > 0:
            print(f"\nActual Train/Val ratio: {summary['train_images']/(summary['train_images']+summary['val_images']):.1%} / {summary['val_images']/(summary['train_images']+summary['val_images']):.1%}")
    
    if summary["errors"]:
        print(f"\nErrors encountered: {len(summary['errors'])}")
        for error in summary["errors"]:
            print(f"  - {error}")
    
    print()
    
    return summary
