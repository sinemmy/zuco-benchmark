"""
It's really hard to figure out the structure of the .mat files from the ZuCo dataset.
This script explores the structure of .mat files from the ZuCo dataset.
It attempts to load the files using both scipy and h5py, and provides detailed information about the contents.
"""

import os
import numpy as np
import scipy.io as sio
import h5py
import sys
from pathlib import Path
import datetime

class OutputCapture:
    """Class to capture output and write to both console and file"""
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log_file = open(output_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def explore_mat_file(filepath):
    """
    Explore the structure of a .mat file to understand its contents.
    """
    print(f"\nExploring file: {filepath}")
    try:
        # Try loading with scipy.io.loadmat
        data = sio.loadmat(filepath)
        print("✓ Loaded with scipy.io.loadmat")
        
        # Print top-level keys
        print(f"Top-level keys: {list(data.keys())}")
        
        # Explore the structure of the data
        for key in data.keys():
            if key.startswith('__'):  # Skip metadata keys
                continue
            
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"\nKey: {key}, Type: {type(value)}, Shape: {value.shape}, Dtype: {value.dtype}")
                
                # If it's a struct array, explore its fields
                if value.dtype.names is not None:
                    print(f"  Fields: {value.dtype.names}")
                    
                    # Look at the first item in more detail if it exists
                    if value.size > 0:
                        first_item = value[0, 0]
                        print(f"  First item type: {type(first_item)}")
                        
                        # If the first item has fields, explore them
                        if hasattr(first_item, 'dtype') and first_item.dtype.names is not None:
                            print(f"  First item fields: {first_item.dtype.names}")
                            
                            # Explore deeper into the structure
                            for field_name in first_item.dtype.names:
                                field_value = first_item[field_name]
                                if isinstance(field_value, np.ndarray):
                                    print(f"    Field: {field_name}, Type: {type(field_value)}, Shape: {field_value.shape}, Dtype: {field_value.dtype}")
                                    
                                    # Look for rawEEG data
                                    if field_name == 'rawEEG' and field_value.size > 0:
                                        print(f"      Found rawEEG data! Shape: {field_value.shape}")
                                        if field_value.dtype.names is not None:
                                            print(f"      rawEEG fields: {field_value.dtype.names}")
                                    
                                    # Look for bands data
                                    if field_name in ['theta', 'alpha', 'beta', 'gamma'] and field_value.size > 0:
                                        print(f"      Found {field_name} band data! Shape: {field_value.shape}")
                                else:
                                    print(f"    Field: {field_name}, Type: {type(field_value)}")
            else:
                print(f"Key: {key}, Type: {type(value)}")
        
    except NotImplementedError:
        # If that fails, try loading with h5py
        print("Could not load with scipy.io.loadmat, trying h5py...")
        try:
            with h5py.File(filepath, 'r') as f:
                print("✓ Loaded with h5py")
                
                # Print top-level keys
                print(f"Top-level keys: {list(f.keys())}")
                
                # Explore the structure
                def explore_group(group, prefix=''):
                    for key in group.keys():
                        item = group[key]
                        path = f"{prefix}/{key}"
                        
                        if isinstance(item, h5py.Group):
                            print(f"Group: {path}")
                            explore_group(item, path)
                        elif isinstance(item, h5py.Dataset):
                            print(f"Dataset: {path}, Shape: {item.shape}, Dtype: {item.dtype}")
                            
                            # Check if this might be rawEEG data
                            if 'rawEEG' in path or ('raw' in path and 'EEG' in path):
                                print(f"  Possible rawEEG data found at {path}")
                
                explore_group(f)
        except Exception as e:
            print(f"Error loading with h5py: {e}")
    
    except Exception as e:
        print(f"Error exploring {filepath}: {e}")


def examine_field(filepath, field_path):
    """
    Examine a specific field or path in the mat file.
    
    Args:
        filepath (str): Path to the .mat file
        field_path (list): List describing the path to the field (e.g., ['data', 0, 'rawEEG'])
    """
    print(f"\nExamining field {field_path} in {filepath}")
    try:
        # Load the mat file
        data = sio.loadmat(filepath)
        
        # Navigate to the specified field
        current = data
        path_str = ""
        
        for i, field in enumerate(field_path):
            path_str += f"['{field}']" if isinstance(field, str) else f"[{field}]"
            
            try:
                if isinstance(field, str):
                    current = current[field]
                else:
                    current = current[field]
                    
                # Print info about the current level
                if isinstance(current, np.ndarray):
                    print(f"Field {path_str}: Type: {type(current)}, Shape: {current.shape}, Dtype: {current.dtype}")
                    
                    # For struct arrays, print the field names
                    if current.dtype.names is not None:
                        print(f"  Fields: {current.dtype.names}")
                else:
                    print(f"Field {path_str}: Type: {type(current)}")
                    
            except (KeyError, IndexError) as e:
                print(f"Error: Could not access {path_str}: {e}")
                return
        
        # If we reached the target field, provide more details
        if isinstance(current, np.ndarray):
            # If it's a small array, print the actual values
            if current.size < 10:
                print(f"Values: {current}")
            else:
                # Otherwise, print some sample values
                flat_values = current.flatten() if hasattr(current, 'flatten') else current
                try:
                    print(f"Sample values (first 5 elements): {flat_values[:5]}")
                except (TypeError, IndexError):
                    print("Could not extract sample values")
                
            # If it might be rawEEG data, print more details
            if 'rawEEG' in str(field_path) or any('rawEEG' in str(f) for f in field_path):
                print("This appears to be rawEEG data. Additional details:")
                if len(current.shape) >= 2:
                    print(f"  Dimensions interpretation: Likely [time_samples, channels]")
                    print(f"  Number of time samples: {current.shape[0]}")
                    print(f"  Number of channels: {current.shape[1] if len(current.shape) > 1 else 1}")
                else:
                    print(f"  Unusual shape for EEG data: {current.shape}")
    
    except Exception as e:
        print(f"Error examining field in {filepath}: {e}")


if __name__ == "__main__":
    # Get the data directory from command line arguments or use a default
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../data/train/"  # Use current directory as default
    
    # Create an output file for logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"zuco_data_exploration_{timestamp}.txt"
    output_path = os.path.join(os.getcwd(), output_file)
    
    # Redirect output to both console and file
    output_capture = OutputCapture(output_path)
    sys.stdout = output_capture
    
    try:
        print(f"ZuCo Data Exploration - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {output_path}")
        print(f"Searching for ZuCo data in: {data_dir}")
        
        # Look for .mat files in the data directory
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        
        if not mat_files:
            print(f"No .mat files found in {data_dir}")
            sys.exit(1)
        
        print(f"Found {len(mat_files)} .mat files in {data_dir}")
        
        # Select and explore the first file
        sample_path = os.path.join(data_dir, mat_files[0])
        print(f"Exploring first file: {sample_path}")
        explore_mat_file(sample_path)
        
        # After the initial exploration, try to find and examine rawEEG data
        print("\nAfter initial exploration, let's examine specific fields...")
        # These field paths will need to be adjusted based on what we find in the initial exploration
        examine_field(sample_path, ['data'])
        
        # Check a second file to see if structure is consistent
        if len(mat_files) > 1:
            print("\nExploring a second file to check consistency")
            sample_path2 = os.path.join(data_dir, mat_files[1])
            explore_mat_file(sample_path2)
            
        print(f"\nExploration complete. Results saved to: {output_path}")
    
    finally:
        # Restore original stdout and close the file
        sys.stdout = sys.__stdout__
        output_capture.close()
        print(f"Exploration complete. Results saved to: {output_path}")