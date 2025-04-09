import h5py
import os
import pickle
import json

def load_matlab_string(matlab_extracted_object):
    """Converts a string loaded from h5py into a python string"""
    return u''.join(chr(c) for c in matlab_extracted_object)

def extract_sentence_content():
    """Extract sentence content from the .mat files"""
    os.makedirs("../portable_data", exist_ok=True)
    
    # Define subjects and directories based on config
    train_subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    train_dir= "../data/train/"

    

    # Dictionary to store sentence content
    sentence_content = {}

        # Combine train and heldout subjects
    for i in range(2):
        if i == 0:
            subjects = train_subjects
            rootdir = train_dir
        else:
            subjects = heldout_subjects
            rootdir = heldout_dir
    
        for subject in subjects:
            sentence_content[subject] = {"NR": {}, "TSR": {}}
            
            # Extract NR sentences
            nr_file = os.path.join(rootdir, f"results{subject}_NR.mat")
            if os.path.exists(nr_file):
                try:
                    with h5py.File(nr_file, 'r') as f:
                        contentData = f['sentenceData']['content']
                        for idx, obj_ref in enumerate(contentData):
                            content_ref = obj_ref[0]
                            sentence = load_matlab_string(f[content_ref])
                            sentence_content[subject]["NR"][idx] = sentence
                    print(f"Extracted {len(sentence_content[subject]['NR'])} NR sentences for {subject}")
                except Exception as e:
                    print(f"Error extracting NR content for {subject}: {str(e)}")
            
            # Extract TSR sentences
            tsr_file = os.path.join(rootdir, f"results{subject}_TSR.mat")
            if os.path.exists(tsr_file):
                try:
                    with h5py.File(tsr_file, 'r') as f:
                        contentData = f['sentenceData']['content']
                        for idx, obj_ref in enumerate(contentData):
                            content_ref = obj_ref[0]
                            sentence = load_matlab_string(f[content_ref])
                            sentence_content[subject]["TSR"][idx] = sentence
                    print(f"Extracted {len(sentence_content[subject]['TSR'])} TSR sentences for {subject}")
                except Exception as e:
                    print(f"Error extracting TSR content for {subject}: {str(e)}")
    
    # Save the sentence content
    with open("../portable_data/sentence_content.json", 'w') as f:
        json.dump(sentence_content, f, indent=2)
    
    print("Sentence content extraction complete!")

if __name__ == "__main__":
    extract_sentence_content()