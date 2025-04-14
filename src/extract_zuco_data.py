"""
This script processes the ZuCo dataset, which contains EEG data and word-level information.
It extracts and organizes the data into a portable format for further analysis. The extracted features include:

Word-level raw EEG data: Extracts EEG data for each word based on the word boundaries
Word boundaries: Preserves the start and end times for each word
EEG frequency band features: Extracts theta, alpha, beta, and gamma band features for each word
Word metadata: Includes sentence context, word position, and subject/task information

The script uses pickle protocol 4 for compatibility with Python 3.11 and organizes the data in a hierarchical structure that makes it easy to access the word-level information you need for mapping to language model embeddings.

When you run this script:
python extract_zuco_data.py /path/to/zuco/data --output-dir /path/to/save/processed/data
It will generate:

Individual files for each subject/task combination (e.g., YAC_NR.pkl)
A combined file with all word data (all_word_data.pkl)
A metadata file with dataset statistics (metadata.json)

Each word entry will contain:

The word itself
Word boundaries (start/end times)
Raw EEG data during the word
Frequency band features
Contextual information (sentence, position)

This structure should give you everything you need for the linear mapping between EEG features and language model embeddings.

to run: 
python src/extract_zuco_data.py data/train/ --output-dir portable_data

"""
import os
import numpy as np
import h5py
import pickle
import json
import re
from tqdm import tqdm
from pathlib import Path

class ZucoDataExtractor:
    """
    A class to extract sentence and word-level EEG data from the ZuCo dataset,
    structured to facilitate mapping with language model embeddings.
    
    This version handles both NR and TSR data formats.
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the ZucoDataExtractor.
        
        Args:
            data_dir (str): Path to the directory containing the ZuCo dataset (.mat files)
            output_dir (str, optional): Path to save the processed data. If None, 
                                        uses data_dir/processed
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'processed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Frequency bands of interest
        self.freq_bands = {
            'theta': ['mean_t1', 'mean_t2'],
            'alpha': ['mean_a1', 'mean_a2'],
            'beta': ['mean_b1', 'mean_b2'],
            'gamma': ['mean_g1', 'mean_g2']
        }
    
    def extract_subject_task_from_filename(self, filename):
        """
        Extract subject ID and task from filename.
        
        Args:
            filename (str): Filename of a ZuCo .mat file (e.g., 'resultsYAC_NR.mat')
            
        Returns:
            tuple: (subject_id, task)
        """
        # Extract subject ID and task using regex
        match = re.match(r'results([A-Z]{3})_([A-Z]{2,3})\.mat', filename)
        if match:
            subject_id = match.group(1)
            task = match.group(2)
            return subject_id, task
        return None, None
    
    def load_matlab_string(self, ref_obj):
        """
        Load a MATLAB string from a reference object.
        
        Args:
            ref_obj: h5py reference object
            
        Returns:
            str: Decoded string
        """
        if ref_obj is None:
            return ""
        
        try:
            # Get the referenced object
            str_obj = ref_obj[()]
            
            # Check if it's bytes or numpy array
            if isinstance(str_obj, bytes):
                return str_obj.decode('utf-8', errors='replace')
            elif isinstance(str_obj, np.ndarray):
                if str_obj.dtype.kind == 'S':  # String type
                    return str(str_obj)
                else:
                    # Try to convert numpy array to bytes
                    try:
                        return str_obj.tobytes().decode('utf-8', errors='replace')
                    except:
                        return str(str_obj)
            else:
                return str(str_obj)
        except Exception as e:
            print(f"Error decoding string: {e}")
            return ""
    
    def get_words_from_sentence(self, h5file, sentence_idx):
        """
        Get words from a sentence, handling both group and dataset structures.
        
        Args:
            h5file: h5py file object
            sentence_idx: Index of the sentence
            
        Returns:
            list: List of words
        """
        words = []
        
        try:
            # Get the reference to the word data
            word_ref = h5file['sentenceData']['word'][sentence_idx, 0]
            
            if isinstance(word_ref, h5py.Reference):
                word_data = h5file[word_ref]
                
                # Handle group structure (for NR data)
                if isinstance(word_data, h5py.Group):
                    # Get all keys in the word group
                    try:
                        word_keys = list(word_data.keys())
                        
                        # Sort keys if they are numeric
                        try:
                            # If keys are numeric, sort them numerically
                            word_keys = sorted(word_keys, key=lambda x: int(x))
                        except:
                            # If keys can't be sorted numerically, use alphabetical order
                            word_keys = sorted(word_keys)
                        
                        # Extract words in order
                        for key in word_keys:
                            try:
                                word_ref = word_data[key]
                                if isinstance(word_ref, h5py.Reference):
                                    word_text = self.load_matlab_string(h5file[word_ref])
                                    words.append(word_text)
                                else:
                                    words.append("")
                            except Exception as e:
                                print(f"Error extracting word with key {key}: {e}")
                                words.append("")
                    except Exception as e:
                        print(f"Error processing word group: {e}")
                
                # Handle dataset structure (for TSR data)
                elif isinstance(word_data, h5py.Dataset):
                    try:
                        # Get the number of words in this dataset
                        word_count = word_data.shape[0] if hasattr(word_data, 'shape') else 0
                        
                        # Extract each word
                        for i in range(word_count):
                            try:
                                word_item_ref = word_data[i, 0]
                                if isinstance(word_item_ref, h5py.Reference):
                                    word_item = h5file[word_item_ref]
                                    word_text = self.load_matlab_string(word_item)
                                    words.append(word_text)
                                else:
                                    words.append("")
                            except Exception as e:
                                print(f"Error extracting word at index {i}: {e}")
                                words.append("")
                    except Exception as e:
                        print(f"Error processing word dataset: {e}")
                
                else:
                    print(f"Unknown word data type: {type(word_data)}")
        
        except Exception as e:
            print(f"Error getting words for sentence {sentence_idx}: {e}")
        
        return words
    
    def get_word_boundaries(self, h5file, sentence_idx, word_count):
        """
        Get word boundaries for a sentence.
        
        Args:
            h5file: h5py file object
            sentence_idx: Index of the sentence
            word_count: Number of words in the sentence
            
        Returns:
            numpy.ndarray: Word boundaries array or None if not available
        """
        try:
            if 'wordbounds' in h5file['sentenceData']:
                wordbounds_ref = h5file['sentenceData']['wordbounds'][sentence_idx, 0]
                if isinstance(wordbounds_ref, h5py.Reference):
                    wordbounds_data = h5file[wordbounds_ref][()]
                    wordbounds = np.asarray(wordbounds_data)
                    
                    # For TSR data, there might be a mismatch between frequency data size and actual words
                    # So we just return what we have and handle alignment elsewhere
                    return wordbounds
        except Exception as e:
            print(f"Error getting word boundaries for sentence {sentence_idx}: {e}")
        
        return None

    def get_frequency_band_data(self, h5file, sentence_idx, word_idx, field):
        """
        Get frequency band data for a specific word.
        
        Args:
            h5file: h5py file object
            sentence_idx: Index of the sentence
            word_idx: Index of the word
            field: Frequency band field name
            
        Returns:
            numpy.ndarray: Frequency band data or None if not available
        """
        try:
            if field in h5file['sentenceData']:
                field_ref = h5file['sentenceData'][field][sentence_idx, 0]
                if isinstance(field_ref, h5py.Reference):
                    field_data = h5file[field_ref][()]
                    field_data_arr = np.asarray(field_data)
                    
                    # Check if we have data for this word
                    if word_idx < len(field_data_arr):
                        word_data = field_data_arr[word_idx]
                        return np.asarray(word_data).astype(np.float32)
        except Exception as e:
            # Silently fail - we'll just skip this feature
            print(f"Error getting {field} data for sentence {sentence_idx}, word {word_idx}: {e}")
        
        return None

    # def get_word_boundaries(self, h5file, sentence_idx, word_count):
    #     """
    #     Get word boundaries for a sentence.
        
    #     Args:
    #         h5file: h5py file object
    #         sentence_idx: Index of the sentence
    #         word_count: Number of words in the sentence
            
    #     Returns:
    #         numpy.ndarray: Word boundaries array or None if not available
    #     """
    #     try:
    #         if 'wordbounds' in h5file['sentenceData']:
    #             wordbounds_ref = h5file['sentenceData']['wordbounds'][sentence_idx, 0]
    #             if isinstance(wordbounds_ref, h5py.Reference):
    #                 wordbounds_data = h5file[wordbounds_ref][()]
    #                 wordbounds = np.asarray(wordbounds_data)
                    
    #                 # Ensure we have boundaries for all words
    #                 if len(wordbounds) >= word_count:
    #                     return wordbounds
    #                 else:
    #                     print(f"Warning: Not enough word boundaries for sentence {sentence_idx}. Got {len(wordbounds)}, need {word_count}")
    #     except Exception as e:
    #         print(f"Error getting word boundaries for sentence {sentence_idx}: {e}")
        
    #     return None
    
    # def get_frequency_band_data(self, h5file, sentence_idx, word_idx, field):
    #     """
    #     Get frequency band data for a specific word.
        
    #     Args:
    #         h5file: h5py file object
    #         sentence_idx: Index of the sentence
    #         word_idx: Index of the word
    #         field: Frequency band field name
            
    #     Returns:
    #         numpy.ndarray: Frequency band data or None if not available
    #     """
    #     try:
    #         if field in h5file['sentenceData']:
    #             field_ref = h5file['sentenceData'][field][sentence_idx, 0]
    #             if isinstance(field_ref, h5py.Reference):
    #                 field_data = h5file[field_ref][()]
    #                 field_data_arr = np.asarray(field_data)
                    
    #                 # Check if we have data for this word
    #                 if word_idx < len(field_data_arr):
    #                     word_data = field_data_arr[word_idx]
    #                     return np.asarray(word_data).astype(np.float32)
    #     except Exception as e:
    #         print(f"Error getting {field} data for sentence {sentence_idx}, word {word_idx}: {e}")
        
    #     return None
    
    def extract_data_from_h5py(self, filepath):
        """
        Extract data from an h5py file, organized by sentences with word-level information.
        
        Args:
            filepath (str): Path to the .mat file
            
        Returns:
            dict: Extracted data organized by sentences
        """
        try:
            with h5py.File(filepath, 'r') as h5file:
                # Extract subject ID and task from filename
                filename = os.path.basename(filepath)
                subject_id, task = self.extract_subject_task_from_filename(filename)
                
                if not subject_id or not task:
                    print(f"Could not extract subject ID and task from {filename}")
                    return None
                
                sentences_data = []
                
                # Check if sentenceData exists
                if 'sentenceData' not in h5file:
                    print("sentenceData not found in file")
                    return None
                
                # Get the number of sentences
                try:
                    sentence_count = h5file['sentenceData']['content'].shape[0]
                    print(f"Found {sentence_count} sentences in {filename}")
                except Exception as e:
                    print(f"Error determining sentence count: {e}")
                    return None
                
                # Iterate through each sentence
                for sentence_idx in tqdm(range(sentence_count), desc=f"Processing {filename}"):
                    try:
                        # Create sentence object
                        sentence_obj = {
                            'subject_id': subject_id,
                            'task': task,
                            'sentence_idx': int(sentence_idx),
                            'text': '',
                            'words': []
                        }
                        
                        # Get sentence content
                        try:
                            content_ref = h5file['sentenceData']['content'][sentence_idx, 0]
                            if isinstance(content_ref, h5py.Reference):
                                content_data = h5file[content_ref]
                                sentence_obj['text'] = self.load_matlab_string(content_data)
                        except Exception as e:
                            print(f"Error getting content for sentence {sentence_idx}: {e}")
                        
                        # Get raw EEG data for the whole sentence
                        raw_sentence_eeg = None
                        try:
                            if 'rawData' in h5file['sentenceData']:
                                rawdata_ref = h5file['sentenceData']['rawData'][sentence_idx, 0]
                                if isinstance(rawdata_ref, h5py.Reference):
                                    raw_data = h5file[rawdata_ref][()]
                                    raw_sentence_eeg = np.asarray(raw_data)
                        except Exception as e:
                            print(f"Error processing raw EEG for sentence {sentence_idx}: {e}")
                        
                        # Get words
                        words = self.get_words_from_sentence(h5file, sentence_idx)
                        if not words:
                            print(f"No words found for sentence {sentence_idx}")
                            continue
                        
                        # Get word boundaries
                        wordbounds = self.get_word_boundaries(h5file, sentence_idx, len(words))
                        
                        # Process each word in the sentence
                        for word_idx, word_text in enumerate(words):
                            try:
                                # Create word object
                                word_obj = {
                                    'word_idx': int(word_idx),
                                    'text': word_text
                                }
                                
                                # Add word boundaries
                                if wordbounds is not None and word_idx < len(wordbounds):
                                    word_obj['boundaries'] = {
                                        'start': float(wordbounds[word_idx, 0]),
                                        'end': float(wordbounds[word_idx, 1])
                                    }
                                
                                # Add raw EEG data for this word
                                if raw_sentence_eeg is not None and 'boundaries' in word_obj:
                                    try:
                                        start_time = word_obj['boundaries']['start']
                                        end_time = word_obj['boundaries']['end']
                                        
                                        # Convert time to sample indices (assuming 500 Hz sampling rate)
                                        start_sample = max(0, int(start_time * 500))
                                        end_sample = min(len(raw_sentence_eeg), int(end_time * 500))
                                        
                                        # Check if indices are valid
                                        if start_sample < end_sample and start_sample >= 0 and end_sample <= len(raw_sentence_eeg):
                                            word_obj['raw_eeg'] = np.asarray(raw_sentence_eeg[start_sample:end_sample, :]).astype(np.float32)
                                    except Exception as e:
                                        print(f"Error extracting raw EEG for word {word_idx} in sentence {sentence_idx}: {e}")
                                
                                # Initialize EEG features
                                word_obj['eeg_features'] = {}
                                
                                # Extract frequency band data for this word
                                for band_name, band_fields in self.freq_bands.items():
                                    word_obj['eeg_features'][band_name] = {}
                                    for field in band_fields:
                                        band_data = self.get_frequency_band_data(h5file, sentence_idx, word_idx, field)
                                        if band_data is not None:
                                            word_obj['eeg_features'][band_name][field] = band_data
                                
                                # Add the processed word to our sentence
                                sentence_obj['words'].append(word_obj)
                            
                            except Exception as e:
                                print(f"Error processing word {word_idx} in sentence {sentence_idx}: {e}")
                        
                        # Add the processed sentence to our collection if it has words
                        if sentence_obj['words']:
                            sentences_data.append(sentence_obj)
                    
                    except Exception as e:
                        print(f"Error processing sentence {sentence_idx}: {e}")
                
                return {
                    'subject_id': subject_id,
                    'task': task,
                    'sentence_count': len(sentences_data),
                    'sentences': sentences_data
                }
        
        except Exception as e:
            print(f"Error loading {filepath} with h5py: {e}")
            return None
    
    def process_all_files(self):
        """
        Process all .mat files in the data directory.
        
        Returns:
            dict: All processed data
        """
        # Find all .mat files in the data directory
        mat_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mat')]
        
        if not mat_files:
            print(f"No .mat files found in {self.data_dir}")
            return None
        
        print(f"Found {len(mat_files)} .mat files")
        
        all_subject_data = {}
        
        # Process each file
        for filename in tqdm(mat_files, desc="Processing files"):
            filepath = os.path.join(self.data_dir, filename)
            processed_data = self.extract_data_from_h5py(filepath)
            
            if processed_data and processed_data['sentences']:
                subject_id = processed_data['subject_id']
                task = processed_data['task']
                
                # Save subject and task specific data
                key = f"{subject_id}_{task}"
                
                # Save with pickle protocol 4 for compatibility with Python 3.11
                subject_task_path = os.path.join(self.output_dir, f"{key}.pkl")
                with open(subject_task_path, 'wb') as f:
                    pickle.dump(processed_data['sentences'], f, protocol=4)
                
                total_words = sum(len(sentence['words']) for sentence in processed_data['sentences'])
                print(f"Saved {len(processed_data['sentences'])} sentences with {total_words} words to {subject_task_path}")
                
                # Add to all subject data
                all_subject_data[key] = processed_data['sentences']
            else:
                print(f"No usable data extracted from {filename}")
        
        if not all_subject_data:
            print("No data was processed.")
            return None
        
        # Save all data combined with pickle protocol 4
        all_data_path = os.path.join(self.output_dir, "all_data.pkl")
        with open(all_data_path, 'wb') as f:
            pickle.dump(all_subject_data, f, protocol=4)
        
        print(f"Saved data for {len(all_subject_data)} subject-task combinations to {all_data_path}")
        
        return all_subject_data
    
    def create_metadata_file(self, all_subject_data):
        """
        Create a metadata file with summary statistics.
        
        Args:
            all_subject_data (dict): Processed data from process_all_files
        """
        if not all_subject_data:
            print("No data to create metadata from.")
            return
        
        # Calculate statistics
        subjects = set()
        tasks = set()
        total_sentences = 0
        total_words = 0
        words_with_raw_eeg = 0
        words_with_boundaries = 0
        sentences_per_subject = {}
        words_per_subject = {}
        sentences_per_task = {}
        words_per_task = {}
        
        # Feature sets for metadata
        eeg_feature_sets = set()
        
        for key, sentences in all_subject_data.items():
            subject_id, task = key.split('_')
            subjects.add(subject_id)
            tasks.add(task)
            
            # Count sentences and words for this subject-task
            sentences_count = len(sentences)
            words_count = sum(len(sentence['words']) for sentence in sentences)
            
            total_sentences += sentences_count
            total_words += words_count
            
            # Update subject and task counts
            if subject_id not in sentences_per_subject:
                sentences_per_subject[subject_id] = 0
                words_per_subject[subject_id] = 0
            if task not in sentences_per_task:
                sentences_per_task[task] = 0
                words_per_task[task] = 0
            
            sentences_per_subject[subject_id] += sentences_count
            words_per_subject[subject_id] += words_count
            sentences_per_task[task] += sentences_count
            words_per_task[task] += words_count
            
            # Count words with raw EEG and boundaries
            for sentence in sentences:
                for word in sentence['words']:
                    if 'raw_eeg' in word:
                        words_with_raw_eeg += 1
                    
                    if 'boundaries' in word:
                        words_with_boundaries += 1
                    
                    # Track EEG feature sets
                    if 'eeg_features' in word:
                        for band_name, band_data in word['eeg_features'].items():
                            for field in band_data.keys():
                                eeg_feature_sets.add(f"{band_name}_{field}")
        
        metadata = {
            "total_sentences": total_sentences,
            "total_words": total_words,
            "words_with_raw_eeg": words_with_raw_eeg,
            "words_with_boundaries": words_with_boundaries,
            "subjects": list(subjects),
            "tasks": list(tasks),
            "sentences_per_subject": sentences_per_subject,
            "words_per_subject": words_per_subject,
            "sentences_per_task": sentences_per_task,
            "words_per_task": words_per_task,
            "available_eeg_features": list(eeg_feature_sets)
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract sentence and word-level EEG data from ZuCo dataset.')
    parser.add_argument('data_dir', help='Path to directory containing ZuCo .mat files')
    parser.add_argument('--output-dir', help='Path to save processed data', default=None)
    
    args = parser.parse_args()
    
    extractor = ZucoDataExtractor(args.data_dir, args.output_dir)
    all_subject_data = extractor.process_all_files()
    
    if all_subject_data:
        extractor.create_metadata_file(all_subject_data)
    else:
        print("No data was processed.")