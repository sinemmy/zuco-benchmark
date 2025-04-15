"""
Diagnostic script to examine ZuCo dataset structure differences between
successfully processed and failing sentences in problematic files.

This script analyzes a single .mat file, extracting and comparing the structure
of sentences that process successfully versus those that fail. The output helps
identify the exact differences that cause extraction errors.

Usage:
python diagnose_zuco_structure.py <path_to_mat_file> <output_file>

Example:
python diagnose_zuco_structure.py data/train/resultsYAC_NR.mat diagnostics/YAC_NR_structure.json
"""

import h5py
import numpy as np
import json
import os
import re
import sys
from pathlib import Path

PROBLEM_INDICES = {
    "YAC_NR": {"good": [0, 1, 2], "bad": [201, 202, 203]},
    "YAC_TSR": {"good": [0, 1, 2], "bad": [171, 172, 173]},
    "YFR_TSR": {"good": [0, 1, 2], "bad": [45, 46, 171]},
    "YLS_NR": {"good": [0, 1, 2], "bad": [45, 46, 47]}
}

# Add this class near the top of your file, after imports
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy types and other special objects."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            # Handle arrays based on their type
            if obj.dtype.kind in ['U', 'S']:  # Unicode or ASCII strings
                return obj.astype(str).tolist() if obj.size > 0 else []
            else:
                return obj.tolist()
        elif isinstance(obj, (bytes, np.bytes_)):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, complex):
            return [obj.real, obj.imag]
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'decode'):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, h5py.Reference):
            return str(obj)
        else:
            try:
                return super().default(obj)
            except:
                return str(obj)

def extract_subject_task_from_filename(filename):
    """Extract subject ID and task from filename."""
    match = re.match(r'results([A-Z]{3})_([A-Z]{2,3})\.mat', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_matlab_string(h5file, ref_obj):
    """Load a MATLAB string from a reference object."""
    if ref_obj is None:
        return ""
    
    try:
        # Get the referenced object
        str_obj = h5file[ref_obj][()]
        
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
        return f"Error decoding string: {e}"

def describe_object(obj, h5file=None, max_items=10):
    """Create a structured description of an h5py object."""
    if isinstance(obj, h5py.Group):
        return {
            "type": "Group",
            "keys": list(obj.keys())[:max_items] + (["..."] if len(obj.keys()) > max_items else []),
            "attrs": dict(obj.attrs)
        }
    elif isinstance(obj, h5py.Dataset):
        data = None
        shape = obj.shape
        dtype = str(obj.dtype)
        attrs = dict(obj.attrs)
        
        # For small datasets, include actual data
        try:
            if len(obj) <= max_items:
                data = obj[()].tolist() if hasattr(obj[()], 'tolist') else str(obj[()])
            else:
                # For larger datasets, provide sample
                if len(shape) == 1:
                    sample = obj[:min(max_items, shape[0])][()]
                    data = sample.tolist() if hasattr(sample, 'tolist') else str(sample)
                elif len(shape) == 2:
                    sample = obj[:min(max_items, shape[0]), :min(max_items, shape[1])][()]
                    data = sample.tolist() if hasattr(sample, 'tolist') else str(sample)
                else:
                    data = f"Array with {len(shape)} dimensions (too complex to show sample)"
        except Exception as e:
            data = f"Error extracting data: {e}"
            
        return {
            "type": "Dataset",
            "shape": shape,
            "dtype": dtype,
            "attrs": attrs,
            "sample_data": data
        }
    elif isinstance(obj, h5py.Reference):
        if h5file is not None:
            try:
                ref_obj = h5file[obj]
                return {
                    "type": "Reference",
                    "target_path": ref_obj.name,
                    "target_type": type(ref_obj).__name__,
                    "description": describe_object(ref_obj, h5file, max_items)
                }
            except Exception as e:
                return {
                    "type": "Reference",
                    "error": f"Error dereferencing: {e}"
                }
        else:
            return {
                "type": "Reference",
                "warning": "No h5file provided to dereference"
            }
    else:
        # Try to convert to a basic Python type
        try:
            if isinstance(obj, np.ndarray):
                if obj.size <= max_items:
                    return {
                        "type": "ndarray",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "data": obj.tolist() if obj.dtype.kind not in ['O'] else str(obj)
                    }
                else:
                    return {
                        "type": "ndarray",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "sample": obj.flatten()[:max_items].tolist() if obj.dtype.kind not in ['O'] else str(obj.flatten()[:max_items])
                    }
            elif isinstance(obj, (int, float, str, bool, list, dict)) or obj is None:
                return obj
            else:
                return str(obj)
        except Exception as e:
            return f"Error describing object: {e}"

def analyze_sentence_structure(h5file, sentence_idx):
    """Analyze the structure of a specific sentence and its data."""
    result = {
        "sentence_idx": sentence_idx,
        "sentence_text": "",
        "structure": {}
    }
    
    try:
        # Get sentence content if available
        try:
            content_ref = h5file['sentenceData']['content'][sentence_idx, 0]
            if isinstance(content_ref, h5py.Reference):
                content_data = h5file[content_ref]
                result["sentence_text"] = load_matlab_string(h5file, content_ref)
        except Exception as e:
            result["sentence_text_error"] = str(e)
            
        # Analyze key components
        components = [
            "word", "wordbounds", "rawData",
            "mean_t1", "mean_t2",  # theta
            "mean_a1", "mean_a2",  # alpha
            "mean_b1", "mean_b2",  # beta
            "mean_g1", "mean_g2"   # gamma
        ]
        
        for component in components:
            try:
                if component in h5file['sentenceData']:
                    comp_ref = h5file['sentenceData'][component][sentence_idx, 0]
                    if isinstance(comp_ref, h5py.Reference):
                        result["structure"][component] = describe_object(h5file[comp_ref], h5file)
                    else:
                        result["structure"][component] = {
                            "type": type(comp_ref).__name__,
                            "value": str(comp_ref)
                        }
                else:
                    result["structure"][component] = "Not found in sentenceData"
            except Exception as e:
                result["structure"][component] = {
                    "error": str(e)
                }
                
        # Try to get word list
        try:
            words = []
            word_ref = h5file['sentenceData']['word'][sentence_idx, 0]
            
            if isinstance(word_ref, h5py.Reference):
                word_data = h5file[word_ref]
                
                # Handle group structure (for NR data)
                if isinstance(word_data, h5py.Group):
                    word_keys = list(word_data.keys())
                    try:
                        word_keys = sorted(word_keys, key=lambda x: int(x))
                    except:
                        word_keys = sorted(word_keys)
                    
                    for key in word_keys[:10]:  # Get first 10 words
                        try:
                            word_item_ref = word_data[key]
                            if isinstance(word_item_ref, h5py.Reference):
                                word_text = load_matlab_string(h5file, word_item_ref)
                                words.append(word_text)
                            else:
                                words.append(f"Non-reference: {type(word_item_ref).__name__}")
                        except Exception as e:
                            words.append(f"Error: {e}")
                
                # Handle dataset structure (for TSR data)
                elif isinstance(word_data, h5py.Dataset):
                    word_count = min(10, word_data.shape[0]) if hasattr(word_data, 'shape') else 0
                    
                    for i in range(word_count):
                        try:
                            word_item_ref = word_data[i, 0]
                            if isinstance(word_item_ref, h5py.Reference):
                                word_text = load_matlab_string(h5file, word_item_ref)
                                words.append(word_text)
                            else:
                                words.append(f"Non-reference: {type(word_item_ref).__name__}")
                        except Exception as e:
                            words.append(f"Error: {e}")
                            
            result["first_10_words"] = words
        except Exception as e:
            result["words_error"] = str(e)
            
        # Try to extract wordbounds details
        try:
            if 'wordbounds' in h5file['sentenceData']:
                wordbounds_ref = h5file['sentenceData']['wordbounds'][sentence_idx, 0]
                if isinstance(wordbounds_ref, h5py.Reference):
                    wordbounds_data = h5file[wordbounds_ref]
                    
                    # Get detailed info about the wordbounds data
                    result["wordbounds_details"] = {
                        "shape": wordbounds_data.shape,
                        "dtype": str(wordbounds_data.dtype),
                        "sample": wordbounds_data[()][:5].tolist() if wordbounds_data.shape[0] > 5 else wordbounds_data[()].tolist()
                    }
                    
                    # Check for specific issue with indexing
                    if len(wordbounds_data.shape) == 2 and wordbounds_data.shape[1] == 1:
                        result["wordbounds_issue"] = "Found single-column array instead of expected two-column array"
        except Exception as e:
            result["wordbounds_error"] = str(e)
            
    except Exception as e:
        result["analysis_error"] = str(e)
        
    return result

def main(input_file, output_file):
    """
    Analyze a ZuCo .mat file to compare successful and failing sentences.
    Writes structured analysis to a JSON output file.
    """
    # Get subject and task from filename
    filename = os.path.basename(input_file)
    subject_id, task = extract_subject_task_from_filename(filename)
    file_key = f"{subject_id}_{task}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        "filename": filename,
        "subject_id": subject_id,
        "task": task,
        "file_analysis": {},
        "sentences": {}
    }
    
    try:
        with h5py.File(input_file, 'r') as h5file:
            # Basic file structure 
            results["file_analysis"]["top_level_keys"] = list(h5file.keys())
            
            if 'sentenceData' in h5file:
                results["file_analysis"]["sentence_data_keys"] = list(h5file['sentenceData'].keys())
                
                # Get sentence count
                try:
                    sentence_count = h5file['sentenceData']['content'].shape[0]
                    results["file_analysis"]["sentence_count"] = int(sentence_count)
                except Exception as e:
                    results["file_analysis"]["sentence_count_error"] = str(e)
                
                # Get indices for this specific file
                if file_key in PROBLEM_INDICES:
                    good_indices = PROBLEM_INDICES[file_key]["good"]
                    bad_indices = PROBLEM_INDICES[file_key]["bad"]
                else:
                    # Default to some reasonable values
                    good_indices = [0, 1, 2]
                    bad_indices = [201, 202, 203]
                
                # Analyze known good sentences
                for i in good_indices:
                    if i < sentence_count:
                        results["sentences"][f"good_{i}"] = analyze_sentence_structure(h5file, i)
                
                # Analyze known problematic sentences
                for i in bad_indices:
                    if i < sentence_count:
                        results["sentences"][f"bad_{i}"] = analyze_sentence_structure(h5file, i)
            else:
                results["file_analysis"]["error"] = "sentenceData not found in file"
    
    except Exception as e:
        results["file_analysis"]["error"] = str(e)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
    
    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diagnose_zuco_structure.py <path_to_mat_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)