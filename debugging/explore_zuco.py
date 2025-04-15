"""
It's really hard to figure out the structure of the .mat files and the format the data is stored 
from the ZuCo dataset.

This script (2/2) explores the structure of .mat files from the ZuCo dataset.
It attempts to load a single mat file and figure out the types in there 
"""

import h5py
import numpy as np
import sys
import os
from pprint import pprint

def explore_sentence_structure(file_path, output_file, sentence_indices=[0, 1, 2], word_indices=[0, 1, 2]):
    """
    Explore the structure of specific sentences and words in the ZuCo dataset.
    
    Args:
        file_path: Path to .mat file
        output_file: File to save output
        sentence_indices: List of sentence indices to explore
        word_indices: List of word indices to explore for each sentence
    """
    with open(output_file, 'w') as out_file:
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Check if sentenceData exists
                if 'sentenceData' not in h5file:
                    out_file.write("sentenceData not found in file\n")
                    return
                
                # Print high-level structure
                out_file.write("\n--- High-level structure ---\n\n")
                for key in h5file.keys():
                    out_file.write(f"/{key} ({type(h5file[key])})\n")
                    if isinstance(h5file[key], h5py.Group):
                        for subkey in list(h5file[key].keys())[:5]:  # Only show first 5 keys
                            out_file.write(f"  /{key}/{subkey} ({type(h5file[key][subkey])})\n")
                        if len(h5file[key]) > 5:
                            out_file.write(f"  ... and {len(h5file[key]) - 5} more keys\n")
                
                # Explore specific sentences and words
                for sentence_idx in sentence_indices:
                    for word_idx in word_indices:
                        out_file.write(f"\n--- Exploring sentence {sentence_idx}, word {word_idx} ---\n\n")
                        
                        # Get sentence content
                        if 'content' in h5file['sentenceData']:
                            out_file.write("\nSentence Content:\n")
                            try:
                                content_dataset = h5file['sentenceData']['content']
                                out_file.write(f"Type: {type(content_dataset)}\n")
                                out_file.write(f"Shape: {content_dataset.shape if hasattr(content_dataset, 'shape') else 'no shape'}\n")
                                out_file.write(f"Size: {content_dataset.size if hasattr(content_dataset, 'size') else 'no size'}\n")
                                
                                if hasattr(content_dataset, 'shape') and sentence_idx < content_dataset.shape[0]:
                                    content_ref = content_dataset[sentence_idx, 0]
                                    out_file.write(f"Content reference type: {type(content_ref)}\n")
                                    
                                    if isinstance(content_ref, h5py.Reference):
                                        content_data = h5file[content_ref]
                                        out_file.write(f"Content data type: {type(content_data)}\n")
                                        out_file.write(f"Content data shape: {content_data.shape if hasattr(content_data, 'shape') else 'no shape'}\n")
                                        
                                        try:
                                            content_bytes = content_data[()]
                                            out_file.write(f"Content bytes type: {type(content_bytes)}\n")
                                            
                                            try:
                                                if hasattr(content_bytes, 'tobytes'):
                                                    content_str = content_bytes.tobytes().decode('utf-8', errors='replace')
                                                    out_file.write(f"Content text: {content_str}\n")
                                                elif isinstance(content_bytes, bytes):
                                                    content_str = content_bytes.decode('utf-8', errors='replace')
                                                    out_file.write(f"Content text (direct bytes): {content_str}\n")
                                                elif isinstance(content_bytes, str):
                                                    out_file.write(f"Content text (direct string): {content_bytes}\n")
                                                else:
                                                    out_file.write(f"Content bytes is type {type(content_bytes)}, dir: {dir(content_bytes)}\n")
                                            except Exception as e:
                                                out_file.write(f"Error decoding content: {e}\n")
                                        except Exception as e:
                                            out_file.write(f"Error accessing content data: {e}\n")
                                else:
                                    out_file.write(f"Sentence index {sentence_idx} out of bounds\n")
                            except Exception as e:
                                out_file.write(f"Error accessing content: {e}\n")
                        
                        # Get words
                        if 'word' in h5file['sentenceData']:
                            out_file.write("\nWords:\n")
                            try:
                                word_dataset = h5file['sentenceData']['word']
                                out_file.write(f"Type: {type(word_dataset)}\n")
                                out_file.write(f"Shape: {word_dataset.shape if hasattr(word_dataset, 'shape') else 'no shape'}\n")
                                
                                if hasattr(word_dataset, 'shape') and sentence_idx < word_dataset.shape[0]:
                                    word_ref = word_dataset[sentence_idx, 0]
                                    out_file.write(f"Word reference type: {type(word_ref)}\n")
                                    
                                    if isinstance(word_ref, h5py.Reference):
                                        word_data = h5file[word_ref]
                                        out_file.write(f"Word data type: {type(word_data)}\n")
                                        out_file.write(f"Word data shape: {word_data.shape if hasattr(word_data, 'shape') else 'no shape'}\n")
                                        out_file.write(f"Word data size: {word_data.size if hasattr(word_data, 'size') else 'no size'}\n")
                                        out_file.write(f"Word data has dir: {dir(word_data)}\n")
                                        
                                        # Try to get a specific word
                                        if hasattr(word_data, 'shape') and word_idx < word_data.shape[0]:
                                            try:
                                                word_item_ref = word_data[word_idx, 0]
                                                out_file.write(f"Word item reference type: {type(word_item_ref)}\n")
                                                
                                                if isinstance(word_item_ref, h5py.Reference):
                                                    word_item = h5file[word_item_ref]
                                                    out_file.write(f"Word item type: {type(word_item)}\n")
                                                    
                                                    try:
                                                        word_bytes = word_item[()]
                                                        out_file.write(f"Word bytes type: {type(word_bytes)}\n")
                                                        out_file.write(f"Word bytes has dir: {dir(word_bytes)}\n")
                                                        
                                                        # Try different methods of decoding
                                                        out_file.write("Trying different decoding methods:\n")
                                                        try:
                                                            if hasattr(word_bytes, 'tobytes'):
                                                                word_str = word_bytes.tobytes().decode('utf-8', errors='replace')
                                                                out_file.write(f"Word text (tobytes): {word_str}\n")
                                                            else:
                                                                out_file.write("No tobytes method available\n")
                                                        except Exception as e:
                                                            out_file.write(f"Error with tobytes decoding: {e}\n")
                                                        
                                                        try:
                                                            if isinstance(word_bytes, bytes):
                                                                word_str = word_bytes.decode('utf-8', errors='replace')
                                                                out_file.write(f"Word text (direct bytes): {word_str}\n")
                                                            else:
                                                                out_file.write(f"Not direct bytes, type is {type(word_bytes)}\n")
                                                        except Exception as e:
                                                            out_file.write(f"Error with direct bytes decoding: {e}\n")
                                                        
                                                        try:
                                                            if isinstance(word_bytes, str):
                                                                out_file.write(f"Word text (already string): {word_bytes}\n")
                                                            else:
                                                                out_file.write(f"Not a string, type is {type(word_bytes)}\n")
                                                        except Exception as e:
                                                            out_file.write(f"Error with string check: {e}\n")
                                                        
                                                        try:
                                                            if isinstance(word_bytes, np.ndarray):
                                                                out_file.write(f"Word bytes is numpy array with dtype: {word_bytes.dtype}\n")
                                                                if word_bytes.dtype.kind == 'S':  # String type
                                                                    word_str = str(word_bytes)
                                                                    out_file.write(f"Word text (numpy string): {word_str}\n")
                                                                else:
                                                                    out_file.write(f"Numpy array but not string type, dtype is {word_bytes.dtype}\n")
                                                            else:
                                                                out_file.write(f"Not a numpy array, type is {type(word_bytes)}\n")
                                                        except Exception as e:
                                                            out_file.write(f"Error with numpy check: {e}\n")
                                                        
                                                        # If it's a tuple, try to inspect it
                                                        try:
                                                            if isinstance(word_bytes, tuple):
                                                                out_file.write(f"Word bytes is a tuple of length {len(word_bytes)}\n")
                                                                for i, item in enumerate(word_bytes[:5]):  # Show only first 5 items
                                                                    out_file.write(f"  Tuple item {i}: {type(item)}, value: {repr(item)}\n")
                                                        except Exception as e:
                                                            out_file.write(f"Error inspecting tuple: {e}\n")
                                                        
                                                    except Exception as e:
                                                        out_file.write(f"Error accessing word item: {e}\n")
                                            except Exception as e:
                                                out_file.write(f"Error accessing word at index {word_idx}: {e}\n")
                                        else:
                                            out_file.write(f"Word index {word_idx} out of bounds or shape not accessible\n")
                                else:
                                    out_file.write(f"Sentence index {sentence_idx} out of bounds for words\n")
                            except Exception as e:
                                out_file.write(f"Error accessing words: {e}\n")
                        
                        # Check one frequency band to understand structure
                        if 'mean_t1' in h5file['sentenceData']:
                            out_file.write("\nFrequency band example (mean_t1):\n")
                            try:
                                field_dataset = h5file['sentenceData']['mean_t1']
                                out_file.write(f"Type: {type(field_dataset)}\n")
                                out_file.write(f"Shape: {field_dataset.shape if hasattr(field_dataset, 'shape') else 'no shape'}\n")
                                
                                if hasattr(field_dataset, 'shape') and sentence_idx < field_dataset.shape[0]:
                                    field_ref = field_dataset[sentence_idx, 0]
                                    if isinstance(field_ref, h5py.Reference):
                                        field_data = h5file[field_ref]
                                        out_file.write(f"Data type: {type(field_data)}\n")
                                        out_file.write(f"Data shape: {field_data.shape if hasattr(field_data, 'shape') else 'no shape'}\n")
                                        
                                        # Check if we have data for the specific word
                                        if hasattr(field_data, 'shape') and word_idx < field_data.shape[0]:
                                            try:
                                                word_field_data = field_data[word_idx]
                                                data_type = type(word_field_data)
                                                data_shape = word_field_data.shape if hasattr(word_field_data, 'shape') else 'no shape'
                                                out_file.write(f"Word {word_idx} data type: {data_type}, shape: {data_shape}\n")
                                                
                                                # If it's a numpy array, show a small sample
                                                if isinstance(word_field_data, np.ndarray) and word_field_data.size > 0:
                                                    sample_size = min(5, word_field_data.size)
                                                    out_file.write(f"Sample data (first {sample_size} elements): {word_field_data.flat[:sample_size]}\n")
                                            except Exception as e:
                                                out_file.write(f"Error accessing data for word {word_idx}: {e}\n")
                            except Exception as e:
                                out_file.write(f"Error exploring frequency band: {e}\n")
                        
                        # List available frequency bands
                        out_file.write("\nAvailable frequency bands:\n")
                        freq_fields = ['mean_t1', 'mean_t2', 'mean_a1', 'mean_a2', 
                                      'mean_b1', 'mean_b2', 'mean_g1', 'mean_g2']
                        for field in freq_fields:
                            if field in h5file['sentenceData']:
                                out_file.write(f"  {field}: Present\n")
                            else:
                                out_file.write(f"  {field}: Not found\n")
        
        except Exception as e:
            out_file.write(f"Error opening file {file_path}: {e}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_zuco.py <path_to_mat_file> [output_file]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "zuco_structure.txt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Exploring {file_path}, saving output to {output_file}")
    
    # Explore a small set of sentences and words
    explore_sentence_structure(
        file_path, 
        output_file, 
        sentence_indices=[0, 1, 7], 
        word_indices=[0, 1, 10]
    )
    
    print(f"Exploration complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()