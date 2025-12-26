import os
import copy
import json
import re
import pprint
import time
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from utils import sectag_to_regex, find_segs, ros_seg, remove_char_json,regexp_ros, return_max_sim_text

# model_folder = "Output_llama3.1_8b"
model_folder = "Output_gemma3_27b"
# model_folder = "Output_mistral-small_24b"
# model_folder = "Output_gpt-oss_20b"

# Add argument for filename and model input
import argparse
parser = argparse.ArgumentParser(description="file and model")

parser.add_argument("--filename", type=str, required=True, default='sample_226')
# parser.add_argument("--model", type=str, required=True, default="llama3.1:8b")
args = parser.parse_args()

if os.path.exists(f"{model_folder}/{args.filename}") == False:
    os.makedirs(f"{model_folder}/{args.filename}")

#Create a new dict to save data - start with reading input file
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


#===========PROMPT to extract diseases and their positive / negative status
dflow_ros_extract = copy.deepcopy(dflow_ros)

"""PROMPT"""
prompt = ChatPromptTemplate([
    ("user", """{ros_text}"""),
    ("user", "convert to json, remove any unnecessary text and make sure the output starts with [")
])

llm1=ChatOllama(
    model="ros_extract"
)

chain = prompt|llm1


for filename in dflow_ros_extract.keys():
    if dflow_ros_extract[filename] != []:
        ros_text = dflow_ros_extract[filename][0]
        ros_extract = chain.invoke(ros_text)
        
        try:
            ros_extract_json = json.loads(ros_extract.content)
        except:
            try:
                # Try again, assuming the error was due to the AI output being not a JSON
                ros_extract2 = llm1.invoke(f"Convert to json, remove any unnecessary text, for example ```, and make sure the output starts with [: {ros_extract.content}")
                ros_extract2_corrected = remove_char_json(ros_extract2.content)
                ros_extract_json = json.loads(ros_extract2_corrected)
            except:
                # If it still fails, just give an empty JSON list
                ros_extract_json = []
        dflow_ros_extract[filename].append(ros_extract_json)
# print(dflow_ros_extract)
#===========PROMPT to identify the systems
dflow_ros_extract_classify = copy.deepcopy(dflow_ros_extract)

llm2=ChatOllama(
    model="ros_classify"
)

for filename in dflow_ros_extract_classify.keys():
    if dflow_ros_extract_classify[filename] != []:
        ros_extracts = dflow_ros_extract_classify[filename][2]
        ai_output = "" # Track the AI output before regex process
        for i in ros_extracts:
            ros_cat = llm2.invoke(i['extract']).content
            ai_output += f'###{ros_cat}\n'
            i['sys']=regexp_ros(ros_cat)
        dflow_ros_extract_classify[filename].append(ai_output)

ros_extract_classify_json = dflow_ros_extract_classify[f"{args.filename}"][2]
print(ros_extract_classify_json)
with open(f"{model_folder}/{args.filename}/output_original.json",'w') as f:
    json.dump(ros_extract_classify_json, f, indent=2)

# Attribution algorithm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

####Record time
start_time = time.perf_counter()


pred_tokens = [item['extract'] for item in ros_extract_classify_json]
pred_tokens_list_encode = [model.encode(item['extract']) for item in ros_extract_classify_json]

pred_tokens_similarity_list = [return_max_sim_text(pred_tokens_encode, clean_original_tokens, original_tokens_seq) 
                               for pred_tokens_encode in pred_tokens_list_encode]

# pred_tokens_similarity_list = parallel_return_max_sim_text(
#     pred_tokens_list_encode, clean_original_tokens, original_tokens_seq)

pred_converted_ = [dict(map(lambda x: (str(x[0]), x[1]), inner)) for inner in pred_tokens_similarity_list]

# Record the end time
end_time = time.perf_counter()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Similarity algorithm time: {elapsed_time}")

###Final prediction processing
pred_extract_converted = []
pred_classify_converted = []
for a, b in zip(pred_converted_, ros_extract_classify_json):
    last_key_extract, last_val_extract = list(b.items())[-2]
    last_key_classify, last_val_classify = list(b.items())[-1]

    extract_dict = a.copy()
    extract_dict[last_key_extract] = last_val_extract
    pred_extract_converted.append(extract_dict)

    convert_dict = a.copy()
    convert_dict[last_key_classify] = last_val_classify
    pred_classify_converted.append(convert_dict)
   
for d in pred_classify_converted:  # rename key to 'status'
    last_key = list(d.keys())[-1]  
    d["status"] = d.pop(last_key)

#Save to json
with open(f"{model_folder}/{args.filename}/output_extract.json", 'w') as f:
    json.dump(pred_extract_converted, f, indent=2)

with open(f"{model_folder}/{args.filename}/output_classify.json", 'w') as f:
    json.dump(pred_classify_converted, f, indent=2)
# if __name__ == "__main__":
