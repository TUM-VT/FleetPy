#!/usr/bin/env python3
"""
Script to convert old pickle training data to compressed pickle format.

This script converts training data files from the old .pkl format to the new 
compressed .pkl format with bz2 compression for reduced storage size.

Usage:
    python convert_training_data.py <input_directory> [output_directory]

If output_directory is not specified, files will be converted in place.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)


def write_compressed_pickle(path: str, data: Any) -> None:
    """
    Write data to a compressed pickle file using bz2 compression.
    
    Args:
        path: File path for the compressed pickle file
        data: Data to write
    """
    try:
        import bz2
        
        with bz2.BZ2File(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        LOG.debug(f"Successfully wrote compressed pickle {path}")
        
    except Exception as e:
        LOG.error(f"Failed to write compressed pickle file {path}: {e}")
        raise


def load_pickle_file(filepath: str) -> Any:
    """Load data from pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        LOG.debug(f"Successfully loaded {filepath}")
        return data
    except Exception as e:
        LOG.error(f"Failed to load pickle file {filepath}: {e}")
        raise


def convert_data_structure(data: Any, filename: str) -> Any:
    """
    Convert old data structure to new expected format.
    
    Args:
        data: Original pickle data
        filename: Name of the file being converted (for format detection)
    
    Returns:
        Data in the new expected format
    """
    try:
        # Detect file type from filename and convert accordingly
        if 'request_features' in filename or 'vehicle_features' in filename:
            # Feature files: should be {id: {feature1: val1, feature2: val2, ...}}
            if isinstance(data, dict):
                # Check if already in new format (nested dict with feature dicts as values)
                sample_value = next(iter(data.values())) if data else None
                if isinstance(sample_value, dict):
                    # Already in new format
                    return data
                else:
                    # Old format - might be flat dict or other structure
                    LOG.warning(f"Unexpected feature data structure in {filename}: {type(sample_value)}")
                    return data
            elif isinstance(data, pd.DataFrame):
                # Convert DataFrame back to nested dict format {id: {features}}
                return data.to_dict('index')
            else:
                LOG.warning(f"Unexpected feature data type in {filename}: {type(data)}")
                return data
                
        elif 'vehicle_request_graph' in filename or 'request_request_graph' in filename:
            # Graph files: should be [{source: id1, target: id2, feature1: val1, ...}, ...]
            if isinstance(data, list):
                # Check if already in new format (list of dicts with source/target)
                if data and isinstance(data[0], dict) and 'source' in data[0]:
                    # Already in new format
                    return data
                else:
                    # Old format - might be different structure
                    LOG.warning(f"Unknown graph data structure in {filename}")
                    return data
            elif isinstance(data, dict):
                # Old format: nested dict {source: {target: features}}
                # Convert to flat list format
                edges = []
                for source, targets in data.items():
                    if isinstance(targets, dict):
                        for target, features in targets.items():
                            edge = {'source': source, 'target': target}
                            
                            if 'request_request_graph' in filename:
                                # RR graph has nested travel_cost structure
                                if isinstance(features, dict) and 'travel_cost' in features:
                                    travel_data = features['travel_cost']
                                    if isinstance(travel_data, dict):
                                        # Flatten the nested travel cost structure
                                        for key, value in travel_data.items():
                                            if isinstance(value, dict):
                                                for subkey, subvalue in value.items():
                                                    edge[f"{key}_{subkey}"] = subvalue
                                            else:
                                                edge[key] = value
                                    else:
                                        edge['travel_cost'] = travel_data
                                else:
                                    # Fallback: add all features
                                    edge.update(features)
                            else:
                                # VR graph has simpler structure
                                if isinstance(features, dict):
                                    edge.update(features)
                                else:
                                    # Features might be a single value (like travel time)
                                    edge['value'] = features
                            edges.append(edge)
                    else:
                        # Targets might be a list or single value
                        if isinstance(targets, (list, tuple)):
                            for target in targets:
                                edges.append({'source': source, 'target': target})
                        else:
                            edges.append({'source': source, 'target': targets})
                return edges
            else:
                LOG.warning(f"Unexpected graph data type in {filename}: {type(data)}")
                return data
                
        elif 'assignment' in filename:
            # Assignment files: keep as-is, usually dict format
            return data
            
        else:
            # Unknown file type, keep as-is
            LOG.debug(f"Unknown file type {filename}, keeping original structure")
            return data
            
    except Exception as e:
        LOG.error(f"Error converting data structure for {filename}: {e}")
        return data


def convert_file_wrapper(args: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
    """
    Wrapper function for convert_file to work with multiprocessing.
    
    Args:
        args: Tuple of (pickle_path, compressed_output_path)
    
    Returns:
        Tuple of (success, filename, error_message)
    """
    pickle_path, compressed_output_path = args
    filename = os.path.basename(pickle_path)
    
    try:
        convert_file(pickle_path, compressed_output_path)
        return True, filename, None
    except Exception as e:
        error_msg = f"Failed to convert {filename}: {e}"
        return False, filename, error_msg


def convert_file(pickle_path: str, compressed_output_path: str) -> None:
    """Convert a single pickle file to compressed pickle format."""
    try:
        # Load pickle data
        data = load_pickle_file(pickle_path)
        
        # Convert old structure to new format if needed
        filename = os.path.basename(pickle_path)
        converted_data = convert_data_structure(data, filename)
        
        # Write compressed pickle format
        write_compressed_pickle(compressed_output_path, converted_data)
        
        # Only log debug level for successful conversions to reduce noise
        LOG.debug(f"Converted: {pickle_path} -> {compressed_output_path}")
        
    except Exception as e:
        LOG.error(f"Failed to convert {pickle_path}: {e}")
        raise


def find_pickle_files(directory: str) -> list:
    """Find all pickle files in directory and subdirectories."""
    pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pickle_files.append(os.path.join(root, file))
    return pickle_files


def filter_files_by_range(pickle_files: list, input_dir: str, start_folder: Optional[int] = None, end_folder: Optional[int] = None) -> list:
    """
    Filter pickle files to only include those in folders within the specified range.
    
    Args:
        pickle_files: List of pickle file paths
        input_dir: Input directory (to calculate relative paths)
        start_folder: Minimum folder number (inclusive)
        end_folder: Maximum folder number (inclusive)
    
    Returns:
        Filtered list of pickle files
    """
    if start_folder is None and end_folder is None:
        return pickle_files
    
    filtered_files = []
    
    for pickle_path in pickle_files:
        # Get relative path from input directory
        rel_path = os.path.relpath(pickle_path, input_dir)
        
        # Extract the first directory component (should be the timestep folder)
        path_parts = rel_path.split(os.sep)
        if len(path_parts) > 1:
            folder_name = path_parts[0]
            
            # Check if folder name is numeric
            try:
                folder_num = int(folder_name)
                
                # Check if within range
                if start_folder is not None and folder_num < start_folder:
                    continue
                if end_folder is not None and folder_num > end_folder:
                    continue
                
                filtered_files.append(pickle_path)
                
            except ValueError:
                # Non-numeric folder name, include it (might be root level files)
                LOG.debug(f"Non-numeric folder '{folder_name}' for file {pickle_path}, including")
                filtered_files.append(pickle_path)
        else:
            # File in root directory, include it
            filtered_files.append(pickle_path)
    
    return filtered_files


def convert_training_data(input_dir: str, output_dir: Optional[str] = None, 
                         start_folder: Optional[int] = None, 
                         end_folder: Optional[int] = None, skip_existing: bool = False,
                         max_workers: Optional[int] = None) -> None:
    """
    Convert all pickle training data files to compressed pickle format using parallel processing.
    
    Args:
        input_dir: Directory containing pickle files
        output_dir: Output directory (if None, converts in place)
        start_folder: Minimum folder number to include (inclusive)
        end_folder: Maximum folder number to include (inclusive)
        skip_existing: If True, skip files that already exist in output directory
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all pickle files
    all_pickle_files = find_pickle_files(input_dir)
    
    # Filter by folder range if specified
    pickle_files = filter_files_by_range(all_pickle_files, input_dir, start_folder, end_folder)
    
    if not pickle_files:
        if start_folder is not None or end_folder is not None:
            range_str = f"folders {start_folder or 'start'}-{end_folder or 'end'}"
            LOG.warning(f"No pickle files found in {input_dir} within {range_str}")
        else:
            LOG.warning(f"No pickle files found in {input_dir}")
        return
    
    range_info = ""
    if start_folder is not None or end_folder is not None:
        range_info = f" (filtered to folders {start_folder or 'start'}-{end_folder or 'end'})"
    
    LOG.info(f"Found {len(pickle_files)} pickle files to convert to compressed pickle format{range_info}")
    
    if len(pickle_files) < len(all_pickle_files):
        LOG.info(f"Filtered out {len(all_pickle_files) - len(pickle_files)} files outside the specified range")
    
    # Set default number of workers based on CPU count
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(pickle_files))
        # Limit to reasonable number to avoid overwhelming the system
        max_workers = min(max_workers, 8)
    
    LOG.info(f"Using {max_workers} parallel workers for conversion")
    
    converted_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Prepare conversion tasks
    conversion_tasks = []
    for pickle_path in pickle_files:
        # Calculate output path for compressed pickle format
        if output_dir:
            # Maintain directory structure in output directory
            rel_path = os.path.relpath(pickle_path, input_dir)
            compressed_output_path = os.path.join(output_dir, rel_path.replace('.pkl', '_compressed.pkl'))
            # Create output directory if needed
            os.makedirs(os.path.dirname(compressed_output_path), exist_ok=True)
        else:
            # Convert in place
            compressed_output_path = pickle_path.replace('.pkl', '_compressed.pkl')
        
        # Check if compressed pickle file already exists and skip if requested
        if skip_existing and os.path.exists(compressed_output_path):
            skipped_count += 1
            LOG.debug(f"Skipped existing file: {compressed_output_path}")
        else:
            conversion_tasks.append((pickle_path, compressed_output_path))
    
    # Convert files in parallel
    if conversion_tasks:
        with tqdm(total=len(conversion_tasks), desc="Converting to compressed pickle", unit="files") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(convert_file_wrapper, task): task 
                    for task in conversion_tasks
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_task):
                    try:
                        success, filename, error_msg = future.result()
                        if success:
                            converted_count += 1
                            pbar.set_postfix_str(f"Completed: {filename}")
                        else:
                            failed_count += 1
                            LOG.error(error_msg)
                            pbar.set_postfix_str(f"Error: {filename}")
                    except Exception as e:
                        task = future_to_task[future]
                        pickle_path = task[0]
                        filename = os.path.basename(pickle_path)
                        failed_count += 1
                        LOG.error(f"Unexpected error processing {filename}: {e}")
                        pbar.set_postfix_str(f"Error: {filename}")
                    
                    # Update progress bar
                    pbar.update(1)
    else:
        LOG.info("No files to convert (all files already exist and --skip-existing was used)")
    
    # Build summary message
    summary_parts = [f"{converted_count} files converted"]
    if skipped_count > 0:
        summary_parts.append(f"{skipped_count} files skipped")
    if failed_count > 0:
        summary_parts.append(f"{failed_count} files failed")
    
    LOG.info(f"Conversion complete: {', '.join(summary_parts)}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python convert_training_data.py <input_directory> [output_directory] [options]")
        print("\nNote: Converts each file to compressed pickle format with bz2 compression")
        print("\nOptions:")
        print("  --start N         Only process folders >= N (folder names must be numeric)")
        print("  --end N           Only process folders <= N (folder names must be numeric)")
        print("  --range N M       Only process folders between N and M (inclusive)")
        print("  --skip-existing   Skip files that already exist in output directory")
        print("  --workers N       Number of parallel workers (default: auto-detect, max 8)")
        print("\nRequirements:")
        print("  pip install tqdm  # For progress bar")
        print("\nExamples:")
        print("  python convert_training_data.py ./data/train")
        print("  python convert_training_data.py ./data/train ./data/train_compressed")
        print("  python convert_training_data.py ./data/train --start 3600 --end 7200")
        print("  python convert_training_data.py ./data/train --range 0 3600")
        print("  python convert_training_data.py ./data/train --skip-existing")
        print("  python convert_training_data.py ./data/train --workers 4")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = None
    start_folder = None
    end_folder = None
    skip_existing = False
    max_workers = None
    
    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--start':
            if i + 1 < len(sys.argv):
                try:
                    start_folder = int(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print(f"Error: --start requires a numeric value")
                    sys.exit(1)
            else:
                print("Error: --start requires a numeric value")
                sys.exit(1)
        elif arg == '--end':
            if i + 1 < len(sys.argv):
                try:
                    end_folder = int(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print(f"Error: --end requires a numeric value")
                    sys.exit(1)
            else:
                print("Error: --end requires a numeric value")
                sys.exit(1)
        elif arg == '--range':
            if i + 2 < len(sys.argv):
                try:
                    start_folder = int(sys.argv[i + 1])
                    end_folder = int(sys.argv[i + 2])
                    i += 2
                except ValueError:
                    print(f"Error: --range requires two numeric values")
                    sys.exit(1)
            else:
                print("Error: --range requires two numeric values")
                sys.exit(1)
        elif arg == '--skip-existing':
            skip_existing = True
        elif arg == '--workers':
            if i + 1 < len(sys.argv):
                try:
                    max_workers = int(sys.argv[i + 1])
                    if max_workers < 1:
                        print(f"Error: --workers must be >= 1")
                        sys.exit(1)
                    i += 1
                except ValueError:
                    print(f"Error: --workers requires a numeric value")
                    sys.exit(1)
            else:
                print("Error: --workers requires a numeric value")
                sys.exit(1)
        elif not output_dir and not arg.startswith('--'):
            output_dir = arg
        elif arg.startswith('--'):
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)
        i += 1
    
    # Validate range
    if start_folder is not None and end_folder is not None and start_folder > end_folder:
        print(f"Error: Start folder ({start_folder}) cannot be greater than end folder ({end_folder})")
        sys.exit(1)
    
    try:
        convert_training_data(input_dir, output_dir, start_folder, end_folder, skip_existing, max_workers)
        LOG.info("All done!")
    except ImportError as e:
        if 'tqdm' in str(e):
            LOG.error("tqdm is required for progress bar. Install with: pip install tqdm")
        else:
            LOG.error(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()