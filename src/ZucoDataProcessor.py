import os
import numpy as np
import scipy.io as sio
import pickle
import json
from tqdm import tqdm
import h5py
import re
from pathlib import Path

class ZucoDataProcessor:
    """
    A class to process the ZuCo 2.0 dataset, extracting word-level EEG features
    and organizing them into a structured format suitable for neural mapping.
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the ZucoDataProcessor.
        
        Args:
            data_dir (str): Path to the directory containing the ZuCo 2.0 dataset
            output_dir (str, optional): Path to save the processed data. If None, 
                                       uses data_dir/processed
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'processed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define paths to the different reading task data
        self.NR_path = os.path.join(data_dir, 'task1-NR')
        self.TSR_path = os.path.join(data_dir, 'task2-SR')
        
        # Initialize lists to store subject IDs and task names
        self.subjects = []
        self.tasks = ['NR', 'TSR']
        
        # Frequency bands of interest
        self.freq_bands = ['theta', 'alpha', 'beta', 'gamma']
        
    def identify_subjects(self):
        """
        Identify subject IDs from the dataset.
        """
        # Check in NR directory for subject IDs
        if os.path.exists(self.NR_path):
            mat_files = [f for f in os.listdir(self.NR_path) if f.endswith('.mat')]
            # Extract subject IDs from filenames using regex
            for file in mat_files:
                match = re.match(r'(?:results|resultsTSR)_([A-Z]{3})\.mat', file)
                if match:
                    subject_id = match.group(1)
                    if subject_id not in self.subjects:
                        self.subjects.append(subject_id)
        
        print(f"Found {len(self.subjects)} subjects: {', '.join(self.subjects)}")
        return self.subjects
    
    def load_mat_file(self, filepath):
        """
        Load a .mat file and return its contents.
        
        Args:
            filepath (str): Path to the .mat file
            
        Returns:
            dict: Contents of the .mat file
        """
        try:
            # Try loading with scipy.io.loadmat (for MATLAB 5.0 format)
            data = sio.loadmat(filepath)
            return data
        except NotImplementedError:
            # If that fails, try loading with h5py (for HDF5 format)
            with h5py.File(filepath, 'r') as f:
                data = {}
                for k, v in f.items():
                    data[k] = np.array(v)
                return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_word_level_data(self, subject_id, task):
        """
        Extract word-level EEG features for a specific subject and task.
        
        Args:
            subject_id (str): Subject ID (e.g., 'YAC')
            task (str): Task name ('NR' or 'TSR')
            
        Returns:
            list: List of dictionaries containing word-level data
        """
        task_dir = self.NR_path if task == 'NR' else self.TSR_path
        result_file = f"results_{subject_id}.mat" if task == 'NR' else f"resultsTSR_{subject_id}.mat"
        filepath = os.path.join(task_dir, result_file)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []
        
        # Load the .mat file
        data = self.load_mat_file(filepath)
        if data is None:
            return []
        
        # Initialize list to store word data
        word_data_list = []
        
        # Process the data structure
        # Note: This part of the code will need to be adjusted based on the 
        # exact structure of the .mat files after exploration
        
        try:
            # Assuming the main data is stored in a field called 'data'
            # This is just a placeholder - will update after exploring actual structure
            if 'data' in data:
                sentences = data['data']
                for sentence_idx, sentence in enumerate(sentences):
                    # Extract sentence-level information
                    sentence_text = self._extract_sentence_text(sentence)
                    
                    # Extract word-level information
                    for word_idx, word_data in enumerate(self._extract_word_data(sentence)):
                        word_text = word_data.get('word', '')
                        word_boundaries = word_data.get('boundaries', {})
                        
                        # Extract EEG features
                        eeg_features = self._extract_eeg_features(sentence, word_idx)
                        
                        # Extract raw EEG data
                        raw_eeg = self._extract_raw_eeg(sentence, word_boundaries)
                        
                        # Compile all information into a word entry
                        word_entry = {
                            'word': word_text,
                            'sentence_id': sentence_idx,
                            'word_position': word_idx,
                            'subject_id': subject_id,
                            'task': task,
                            'eeg_features': eeg_features,
                            'raw_eeg': raw_eeg,
                            'word_boundaries': word_boundaries
                        }
                        
                        word_data_list.append(word_entry)
        
        except Exception as e:
            print(f"Error processing data for {subject_id}, {task}: {e}")
            
        return word_data_list
    
    def _extract_sentence_text(self, sentence):
        """
        Extract the text of a sentence from the sentence data.
        This is a placeholder method and will need to be updated based on actual data structure.
        """
        # Placeholder implementation
        return "Sample sentence text"
    
    def _extract_word_data(self, sentence):
        """
        Extract word-level data from a sentence.
        This is a placeholder method and will need to be updated based on actual data structure.
        """
        # Placeholder implementation
        return [{'word': 'sample', 'boundaries': {'start_time': 0, 'end_time': 100}}]
    
    def _extract_eeg_features(self, sentence, word_idx):
        """
        Extract EEG features for a specific word.
        This is a placeholder method and will need to be updated based on actual data structure.
        """
        # Placeholder implementation
        return {band: np.random.rand(105) for band in self.freq_bands}
    
    def _extract_raw_eeg(self, sentence, word_boundaries):
        """
        Extract raw EEG data for a specific word.
        This is a placeholder method and will need to be updated based on actual data structure.
        """
        # Placeholder implementation
        # Simulating 500 Hz sampling for a 200ms window (100 samples) with 105 electrodes
        return np.random.rand(100, 105)
    
    def process_all_data(self):
        """
        Process data for all subjects and tasks, saving the results.
        """
        # First, identify all subjects
        self.identify_subjects()
        
        all_word_data = []
        
        # Process each subject and task
        for subject_id in tqdm(self.subjects, desc="Processing subjects"):
            for task in self.tasks:
                word_data = self.extract_word_level_data(subject_id, task)
                all_word_data.extend(word_data)
                
                # Save subject and task specific data
                subject_task_path = os.path.join(self.output_dir, f"{subject_id}_{task}.pkl")
                with open(subject_task_path, 'wb') as f:
                    pickle.dump(word_data, f)
                print(f"Saved {len(word_data)} word entries to {subject_task_path}")
        
        # Save all data combined
        all_data_path = os.path.join(self.output_dir, "all_word_data.pkl")
        with open(all_data_path, 'wb') as f:
            pickle.dump(all_word_data, f)
        print(f"Saved {len(all_word_data)} total word entries to {all_data_path}")
        
        return all_word_data
    
    def create_metadata_file(self, all_word_data):
        """
        Create a metadata file with summary statistics.
        
        Args:
            all_word_data (list): List of all processed word data
        """
        if not all_word_data:
            print("No word data to create metadata from.")
            return
            
        metadata = {
            "total_words": len(all_word_data),
            "subjects": self.subjects,
            "tasks": self.tasks,
            "words_per_subject": {subject: sum(1 for item in all_word_data if item['subject_id'] == subject) 
                                 for subject in self.subjects},
            "words_per_task": {task: sum(1 for item in all_word_data if item['task'] == task) 
                              for task in self.tasks},
            "feature_dimensions": {
                "eeg_features": {band: all_word_data[0]['eeg_features'][band].shape if band in all_word_data[0]['eeg_features'] else None 
                                for band in self.freq_bands},
                "raw_eeg": all_word_data[0]['raw_eeg'].shape if 'raw_eeg' in all_word_data[0] else None
            }
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")


# Debug function to explore the structure of a mat file
def explore_mat_file(filepath):
    """
    Explore the structure of a .mat file to understand its contents.
    """
    try:
        # Try loading with scipy.io.loadmat
        data = sio.loadmat(filepath)
        print("Loaded with scipy.io.loadmat")
        
        # Print top-level keys
        print(f"Top-level keys: {list(data.keys())}")
        
        # Explore the structure of the data
        for key in data.keys():
            if key.startswith('__'):  # Skip metadata keys
                continue
            
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}, Dtype: {value.dtype}")
                
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
                                    
                                    # Check if this field contains rawEEG data
                                    if field_name == 'rawEEG' and field_value.size > 0:
                                        print(f"      Found rawEEG data! Shape: {field_value.shape}")
                                        if field_value.dtype.names is not None:
                                            print(f"      rawEEG fields: {field_value.dtype.names}")
                                else:
                                    print(f"    Field: {field_name}, Type: {type(field_value)}")
            else:
                print(f"Key: {key}, Type: {type(value)}")
        
    except NotImplementedError:
        # If that fails, try loading with h5py
        print("Could not load with scipy.io.loadmat, trying h5py...")
        with h5py.File(filepath, 'r') as f:
            print("Loaded with h5py")
            
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
        print(f"Error exploring {filepath}: {e}")


# Function to further examine a specific field in the mat file
def examine_field(filepath, field_path):
    """
    Examine a specific field or path in the mat file.
    
    Args:
        filepath (str): Path to the .mat file
        field_path (list): List describing the path to the field (e.g., ['data', 0, 'rawEEG'])
    """
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
                print(f"Sample values (first 5 elements): {current.flat[:5]}")
                
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
        print(f"Error examining field {field_path} in {filepath}: {e}")


if __name__ == "__main__":
    # Define the path to your ZuCo 2.0 dataset
    data_dir = "/path/to/zuco/data"  # Update this to your actual data path
    
    # First, let's explore a sample mat file to understand its structure
    nr_dir = os.path.join(data_dir, 'task1-NR')
    if os.path.exists(nr_dir):
        sample_files = [f for f in os.listdir(nr_dir) if f.endswith('.mat')]
        if sample_files:
            sample_path = os.path.join(nr_dir, sample_files[0])
            print(f"Exploring sample file: {sample_path}")
            explore_mat_file(sample_path)
            
            # After exploring, if we know the path to rawEEG, examine it
            # This will need to be adjusted based on the actual structure
            # examine_field(sample_path, ['data', 0, 'rawEEG'])
    
    # Initialize and run the processor
    processor = ZucoDataProcessor(data_dir)
    all_word_data = processor.process_all_data()
    processor.create_metadata_file(all_word_data)