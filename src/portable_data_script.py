import numpy as np
import os
import pickle
import json
import sys

# Import their code
import config
import data_helpers as dh

def extract_portable_dataset():
    """Extract features and save in a portable format."""
    # Create output directory
    os.makedirs("portable_data", exist_ok=True)
    
    # Load features using their code
    train = dh.get_or_extract_features(config.subjects, config.rootdir)
    
    # For each feature set, save in portable format
    for feature_set in train['features']:
        # Create a dictionary with all required information
        portable_data = {
            'features': train['features'][feature_set],
            'labels': train['labels'][feature_set],
            'idxs': train['idxs'][feature_set] if 'idxs' in train else None
        }
        
        # Save as pickle (more reliable for numpy arrays)
        with open(f"portable_data/{feature_set}.pkl", 'wb') as f:
            pickle.dump(portable_data, f)
            
        print(f"Saved portable data for {feature_set}")
    
    # Save config information
    with open("portable_data/config_info.json", 'w') as f:
        json.dump({
            'subjects': config.subjects,
            'feature_sets': config.feature_sets
        }, f)
    
    print("Extraction complete!")

if __name__ == "__main__":
    extract_portable_dataset()