import os
import copy
import json
import re
import pprint
import pandas as pd
from utils import sectag_to_regex, find_segs, ros_seg
# Add argument for filename
import argparse
parser = argparse. ArgumentParser(description="file and model")

parser.add_argument("--filename", type=str, required=True, default='sample_226')
# parser.add_argument("--model", type=str, required=True, default="llama3.1:8b")
args = parser.parse_args()

if os.path.exists(f"input/{args.filename}") == False:
    os.makedirs(f"input/{args.filename}")

## read the original label
input = pd.read_csv('input_all.csv')
input_filename = input[input["file"]==f"{args.filename}.txt"]
# print(input_filename)
label_extract = input_filename['status'].to_list()
label_classify = input_filename['sys'].to_list()

phrases = input_filename['phrases'].to_list()

"""Extract the original text in ROS segment"""
dflow = {}

with open(f"100notes_txt/{args.filename}.txt") as f:
    dflow[args.filename]=f.read()

#Extract the ROS segment
dflow_ros = {}
header_patterns, seg_names = sectag_to_regex(r'SecTag.csv', 'kmname', 'str')
for filename, note in dflow.items():
    segs = find_segs(note, header_patterns, seg_names)
    ros_data = ros_seg(note, segs)        
    dflow_ros[filename] = ros_data

if args.filename == 'sample_365':
    dflow_ros[f"{args.filename}"][0] = 'REVIEW OF SYSTEMS:\nGeneral:  Negative.\nHEENT:  She does complain of some allergies, sneezing, and sore throat.  She wears glasses.\nPulmonary history:  She has bit of a cough with her allergies.\nCardiovascular history:  Negative for chest pain or palpitations.  She does have hypertension.\nGI history:  Negative for abdominal pain or blood in the stool.\nGU history:  Negative for dysuria or frequency.  ' \
    'She empties okay.\nNeurologic history:  Positive for paresthesias to the toes of both feet, worse on the right.\n' \
    'Musculoskeletal history:  Positive for shoulder pain.\nPsychiatric history:  Positive for insomnia.\nDermatologic history:  Positive for a spot on her right cheek, which she was afraid was a precancerous condition.\nMetabolic history:  She has hypothyroidism.\nHematologic history:  Positive for essential thrombocythemia and anemia.'

#Removing special characters
text = dflow_ros[f"{args.filename}"][0]
clean_text = re.sub(r'[.,:]', '', text)
clean_original_tokens = clean_text.split()
original_tokens_seq = [(i, token) for i, token in enumerate(clean_original_tokens)]


with open(f"ros_text/{args.filename}.txt", "w") as f: #Write ros text to file
    f.write(text + "\n")
    f.write(str(clean_original_tokens) + "\n")
    f.write(str(original_tokens_seq) + "\n")

# Mapping label to original paragraph with text sequence

index_to_word = {i: w for i, w in original_tokens_seq}
words = [w for _, w in original_tokens_seq]
# print(index_to_word)
# Label for negation extraction
label_extract_converted = []

# tokens = [(int(idx), word) for idx, word in index_to_word.items()]
# tokens.sort(key=lambda x: x[0])  # ensure order by index

# results = {}
# for phrase in phrases:
#     phrase_tokens = phrase.split()
#     matches = []
    
#     # sliding window search for multiple matches
#     for i in range(len(tokens) - len(phrase_tokens) + 1):
#         window = tokens[i:i+len(phrase_tokens)]
#         window_words = [w for _, w in window]
        
#         if window_words == phrase_tokens:
#             matches.append(window)
    
#     results[phrase] = matches
# print(len(phrases))


for phrase in phrases:
    phrase_tokens = phrase.split()
    length = len(phrase_tokens)
    # Search for sequence in word list
    for i in range(len(words) - length + 1):
        if words[i:i+length] == phrase_tokens:
            # Build dict {index: word}
            seq_dict = {str(idx): index_to_word[idx] 
                        for idx in range(i, i+length)}
            if seq_dict.keys() not in [key for l in label_extract_converted for key in l.keys()]: ##avoid matching again if words appear twice
                label_extract_converted.append(seq_dict)
            break
for k, v in zip(label_extract_converted, label_extract): #insert label:
    k["status"] = v

# print(label_extract_converted)

# Label for classification
label_classify_converted = []
for phrase in phrases:
    phrase_tokens = phrase.split()
    length = len(phrase_tokens)

    # Search for sequence in word list
    for i in range(len(words) - length + 1):
        if words[i:i+length] == phrase_tokens:
            # Build dict {index: word}
            seq_dict = {str(idx): index_to_word[idx] 
                        for idx in range(i, i+length)}
            label_classify_converted.append(seq_dict)
            break
#insert label:
for k, v in zip(label_classify_converted, label_classify):
    k["status"] = v


#Write to file
with open(f'input/{args.filename}/label_extract.json','w') as f:
    json.dump(label_extract_converted,f,indent=2)

with open(f'input/{args.filename}/label_classify.json','w') as f:
    json.dump(label_classify_converted,f,indent=2)

