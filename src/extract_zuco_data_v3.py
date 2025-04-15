"""
Enhanced ZuCo data extractor that builds on extract_zuco_data_v2.py
with improved error handling and logging of problematic sentences/words.

This script processes the ZuCo dataset, extracts word-level EEG data and 
boundaries, and properly logs any problematic data for later reference.

Usage:
python extract_zuco_data_enhanced.py data/train/ --output-dir portable_data
"""
import os
import numpy as np
import h5py
import pickle
import json
import re
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zuco_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ZucoExtractor")

class ZucoDataExtractor:
    """
    Enhanced version of ZucoDataExtractor with improved error handling
    and logging of problematic sentences/words.
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
        
        # Initialize tracking for problematic data
        self.problematic_data = {
            "bad_sentences": {},  # Subject_Task -> [sentence_indices]
            "bad_words": {},      # Subject_Task -> {sentence_idx: [word_indices]}
            "stats": {
                "total_sentences": 0,
                "skipped_sentences": 0,
                "total_words": 0,
                "skipped_words": 0
            }
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
            logger.debug(f"Error decoding string: {e}")
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
        bad_word_indices = []
        
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
                        for word_idx, key in enumerate(word_keys):
                            try:
                                word_ref = word_data[key]
                                if isinstance(word_ref, h5py.Reference):
                                    word_text = self.load_matlab_string(h5file[word_ref])
                                    words.append(word_text)
                                else:
                                    logger.warning(f"Non-reference word data in sentence {sentence_idx}, word {word_idx}")
                                    words.append("")
                                    bad_word_indices.append(word_idx)
                            except Exception as e:
                                logger.warning(f"Error extracting word with key {key}: {e}")
                                words.append("")
                                bad_word_indices.append(word_idx)
                    except Exception as e:
                        logger.warning(f"Error processing word group: {e}")
                
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
                                    logger.warning(f"Non-reference word data in sentence {sentence_idx}, word {i}")
                                    words.append("")
                                    bad_word_indices.append(i)
                            except Exception as e:
                                logger.warning(f"Error extracting word at index {i}: {e}")
                                words.append("")
                                bad_word_indices.append(i)
                    except Exception as e:
                        logger.warning(f"Error processing word dataset: {e}")
                
                else:
                    logger.warning(f"Unknown word data type: {type(word_data)}")
            else:
                logger.warning(f"Word data for sentence {sentence_idx} is not a reference")
        
        except Exception as e:
            logger.warning(f"Error getting words for sentence {sentence_idx}: {e}")
        
        return words, bad_word_indices
    
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
                    
                    # Check if this is a problematic 1x1 array with NaN
                    if wordbounds.shape == (1, 1) and np.isnan(wordbounds[0, 0]):
                        logger.warning(f"Problematic wordbounds for sentence {sentence_idx} (1x1 NaN array)")
                        return None
                    
                    # Check if it has the expected number of columns
                    if len(wordbounds.shape) != 2 or wordbounds.shape[0] < 2:
                        logger.warning(f"Unexpected wordbounds shape for sentence {sentence_idx}: {wordbounds.shape}")
                        return None
                    
                    return wordbounds
        except Exception as e:
            logger.warning(f"Error getting word boundaries for sentence {sentence_idx}: {e}")
        
        return None

    def get_frequency_band_data(self, h5file, sentence_idx, word_idx, field):
        """
        Get frequency band data for a specific word, with improved error handling.
        
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
                    
                    # Check if this is a problematic 1x1 array with NaN
                    if field_data_arr.shape == (1, 1) and np.isnan(field_data_arr[0, 0]):
                        return None
                    
                    # Different handling based on data shape
                    if len(field_data_arr.shape) == 1:
                        # If it's a 1D array, it might be a single value per electrode
                        if word_idx == 0:  # Only use for the first word
                            return field_data_arr.astype(np.float32)
                        return None
                    elif len(field_data_arr.shape) == 2:
                        # If 2D array check if word_idx is valid
                        if word_idx < field_data_arr.shape[0]:
                            word_data = field_data_arr[word_idx]
                            return word_data.astype(np.float32)
                        else:
                            # If the word index is out of bounds, check if transposed
                            if field_data_arr.shape[1] > word_idx:
                                word_data = field_data_arr[:, word_idx]
                                return word_data.astype(np.float32)
                            else:
                                logger.debug(f"Word index {word_idx} out of bounds for {field} in sentence {sentence_idx}. Shape: {field_data_arr.shape}")
                    else:
                        # More complex structure, try the first dimension
                        if field_data_arr.shape[0] > word_idx:
                            try:
                                # Try to extract data for this word
                                word_data = field_data_arr[word_idx]
                                return np.asarray(word_data).astype(np.float32)
                            except Exception as e:
                                logger.debug(f"Error extracting {field} data for word {word_idx} in sentence {sentence_idx}: {e}")
                                # Return the whole array if individual access fails
                                return field_data_arr.astype(np.float32)
                        else:
                            logger.debug(f"Word index {word_idx} out of bounds for {field} in sentence {sentence_idx}. Shape: {field_data_arr.shape}")
                else:
                    # Only print once per sentence to avoid spam
                    if word_idx == 0:
                        logger.debug(f"Field {field} not found in sentence {sentence_idx}")
        except Exception as e:
            logger.debug(f"Error getting {field} data for sentence {sentence_idx}, word {word_idx}: {e}")
        
        return None

    
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
                    logger.error(f"Could not extract subject ID and task from {filename}")
                    return None
                
                subject_task_key = f"{subject_id}_{task}"
                sentences_data = []
                bad_sentences = []
                bad_words = {}
                
                # Check if sentenceData exists
                if 'sentenceData' not in h5file:
                    logger.error("sentenceData not found in file")
                    return None
                
                # Get the number of sentences
                try:
                    sentence_count = h5file['sentenceData']['content'].shape[0]
                    logger.info(f"Found {sentence_count} sentences in {filename}")
                    self.problematic_data["stats"]["total_sentences"] += sentence_count
                except Exception as e:
                    logger.error(f"Error determining sentence count: {e}")
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
                            logger.warning(f"Error getting content for sentence {sentence_idx}: {e}")
                        
                        # Get raw EEG data for the whole sentence
                        raw_sentence_eeg = None
                        try:
                            if 'rawData' in h5file['sentenceData']:
                                rawdata_ref = h5file['sentenceData']['rawData'][sentence_idx, 0]
                                if isinstance(rawdata_ref, h5py.Reference):
                                    raw_data = h5file[rawdata_ref][()]
                                    raw_sentence_eeg = np.asarray(raw_data)
                                    
                                    # Check if this is a problematic NaN array
                                    if raw_sentence_eeg.shape == (1, 1) and np.isnan(raw_sentence_eeg[0, 0]):
                                        logger.warning(f"Problematic rawData for sentence {sentence_idx} (NaN array)")
                                        raw_sentence_eeg = None
                                        bad_sentences.append(sentence_idx)
                        except Exception as e:
                            logger.warning(f"Error processing raw EEG for sentence {sentence_idx}: {e}")
                        
                        # Get words
                        words, bad_word_indices = self.get_words_from_sentence(h5file, sentence_idx)
                        if bad_word_indices:
                            bad_words[sentence_idx] = bad_word_indices
                            
                        if not words:
                            logger.warning(f"No words found for sentence {sentence_idx}")
                            bad_sentences.append(sentence_idx)
                            self.problematic_data["stats"]["skipped_sentences"] += 1
                            continue
                        
                        # Get word boundaries
                        wordbounds = self.get_word_boundaries(h5file, sentence_idx, len(words))
                        if wordbounds is None:
                            logger.warning(f"No valid word boundaries for sentence {sentence_idx}")
                            bad_sentences.append(sentence_idx)
                            self.problematic_data["stats"]["skipped_sentences"] += 1
                            continue
                        
                        # Process each word in the sentence
                        for word_idx, word_text in enumerate(words):
                            try:
                                # Create word object
                                word_obj = {
                                    'word_idx': int(word_idx),
                                    'text': word_text
                                }
                                
                                # Add word boundaries
                                try:
                                    if wordbounds is not None and word_idx < wordbounds.shape[1]:
                                        word_obj['boundaries'] = {
                                            'start': float(wordbounds[0, word_idx]),
                                            'end': float(wordbounds[2, word_idx])
                                        }
                                    else:
                                        # Skip words without boundaries
                                        logger.debug(f"Word {word_idx} in sentence {sentence_idx} has no boundaries")
                                        if sentence_idx not in bad_words:
                                            bad_words[sentence_idx] = []
                                        bad_words[sentence_idx].append(word_idx)
                                        self.problematic_data["stats"]["skipped_words"] += 1
                                        continue
                                except Exception as e:
                                    logger.warning(f"Error setting boundaries for word {word_idx} in sentence {sentence_idx}: {e}")
                                    if sentence_idx not in bad_words:
                                        bad_words[sentence_idx] = []
                                    bad_words[sentence_idx].append(word_idx)
                                    self.problematic_data["stats"]["skipped_words"] += 1
                                    continue
                                
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
                                        logger.debug(f"Error extracting raw EEG for word {word_idx} in sentence {sentence_idx}: {e}")
                                
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
                                self.problematic_data["stats"]["total_words"] += 1
                            
                            except Exception as e:
                                logger.warning(f"Error processing word {word_idx} in sentence {sentence_idx}: {e}")
                                if sentence_idx not in bad_words:
                                    bad_words[sentence_idx] = []
                                bad_words[sentence_idx].append(word_idx)
                                self.problematic_data["stats"]["skipped_words"] += 1
                        
                        # Add the processed sentence to our collection if it has words
                        if sentence_obj['words']:
                            sentences_data.append(sentence_obj)
                    
                    except Exception as e:
                        logger.warning(f"Error processing sentence {sentence_idx}: {e}")
                        bad_sentences.append(sentence_idx)
                        self.problematic_data["stats"]["skipped_sentences"] += 1
                
                # Store problematic data
                if bad_sentences:
                    self.problematic_data["bad_sentences"][subject_task_key] = bad_sentences
                
                if bad_words:
                    self.problematic_data["bad_words"][subject_task_key] = bad_words
                
                return {
                    'subject_id': subject_id,
                    'task': task,
                    'sentence_count': len(sentences_data),
                    'sentences': sentences_data
                }
        
        except Exception as e:
            logger.error(f"Error loading {filepath} with h5py: {e}")
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
            logger.error(f"No .mat files found in {self.data_dir}")
            return None
        
        logger.info(f"Found {len(mat_files)} .mat files")
        
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
                logger.info(f"Saved {len(processed_data['sentences'])} sentences with {total_words} words to {subject_task_path}")
                
                # Add to all subject data
                all_subject_data[key] = processed_data['sentences']
            else:
                logger.warning(f"No usable data extracted from {filename}")
        
        if not all_subject_data:
            logger.error("No data was processed.")
            return None
        
        # Save all data combined with pickle protocol 4
        all_data_path = os.path.join(self.output_dir, "all_data.pkl")
        with open(all_data_path, 'wb') as f:
            pickle.dump(all_subject_data, f, protocol=4)
        
        # Save problematic data log
        problem_log_path = os.path.join(self.output_dir, "problematic_data.json")
        with open(problem_log_path, 'w') as f:
            json.dump(self.problematic_data, f, indent=2)
        
        logger.info(f"Saved data for {len(all_subject_data)} subject-task combinations to {all_data_path}")
        logger.info(f"Saved problematic data log to {problem_log_path}")
        
        # Log statistics
        logger.info(f"Extraction statistics:")
        logger.info(f"  Total sentences: {self.problematic_data['stats']['total_sentences']}")
        logger.info(f"  Skipped sentences: {self.problematic_data['stats']['skipped_sentences']}")
        logger.info(f"  Total words: {self.problematic_data['stats']['total_words']}")
        logger.info(f"  Skipped words: {self.problematic_data['stats']['skipped_words']}")
        
        return all_subject_data
    
    def create_metadata_file(self, all_subject_data):
        """
        Create a metadata file with summary statistics.
        
        Args:
            all_subject_data (dict): Processed data from process_all_files
        """
        if not all_subject_data:
            logger.error("No data to create metadata from.")
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
            "available_eeg_features": list(eeg_feature_sets),
            "problematic_data_stats": self.problematic_data["stats"]
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")


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
        logger.error("No data was processed.")