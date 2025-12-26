import os
import json
import re
import pprint
import pandas as pd
from utils import drop_last_pair, return_original_dict, extract_labels_preds, exact_match_metrics, relax_match_metrics

# list of file
list_file =("sample_223",
 "sample_225",
 "sample_226",
 "sample_343",
 "sample_365",
 "sample_377",
 "sample_388",
 "sample_391",
 "sample_393",
 "sample_398",
 "sample_402",
 "sample_476",
 "sample_576",
 "sample_1248",
 "sample_1252",
 "sample_1419",
 "sample_1495",
 "sample_1505",
 "sample_1592",
 "sample_1921",
 "sample_2746",
 "sample_2780",
 "sample_2789",
 "sample_2790")
# Add argument for filename and model input
# import argparse
# parser = argparse.ArgumentParser(description="file")
# parser.add_argument("--filename", type=str, required=False, default='sample_226')
# args = parser.parse_args()

# model_no_sim_folder = "Output_llama3.1_8b_no_sim"
# model_no_sim_folder = "Output_gemma3_27b_no_sim"
# model_no_sim_folder = "Output_mistral-small_24b_no_sim"
model_no_sim_folder = "Output_gpt-oss_20b_no_sim"


"""
DISEASES EXTRACTION AND NEGATION DETECTION PERFORMANCE
"""
for sample in list_file:
# Read the label and prediction files
    with open(f"input/{sample}/label_extract.json",'r') as f:
        label_converted = json.load(f)

    with open(f"{model_no_sim_folder}/{sample}/output_extract.json",'r') as file:
        pred_converted = json.load(file)

    print(f"RESULT FOR FILE {sample}")

    #Extract evaluation label - prediction pairs - EM, RM, OD, UD
    em_labels, em_preds, rm_labels, rm_preds, od, ud = extract_labels_preds(label_converted, pred_converted)

    ## EXACT MATCH metrics
    precision_em, sensitivity_em, f1_em, em_ex_tp, em_ex_fp, em_ex_ud, em_ex_od = exact_match_metrics(em_labels, em_preds, ud, rm_preds, od)
    print(f"Extraction exact match performance: precision {precision_em}, sensitivity {sensitivity_em}, f1 score {f1_em}")
    ## RELAX MATCH metrics
    precision_rm, sensitivity_rm, f1_rm, rm_ex_tp, rm_ex_fp, rm_ex_ud, rm_ex_od = relax_match_metrics(em_labels, em_preds, ud, rm_labels, rm_preds, od)
    print(f"Extraction relax match performance: precision {precision_rm}, sensitivity {sensitivity_rm}, f1 score {f1_rm}")

    """
    CLASSIFYING TO BODY SYSTEMS PERFORMANCE
    """
    # Read the label and prediction files
    with open(f"input/{sample}/label_classify.json",'r') as f:
        label_converted = json.load(f)

    with open(f"{model_no_sim_folder}/{sample}/output_classify.json",'r') as file:
        pred_converted = json.load(file)

    #Extract evaluation label - prediction pairs - EM, RM, OD, UD
    em_labels, em_preds, rm_labels, rm_preds, od, ud = extract_labels_preds(label_converted, pred_converted)

    ## EXACT MATCH metrics
    precision_em, sensitivity_em, f1_em, em_cl_tp, em_cl_fp, em_cl_ud, em_cl_od = exact_match_metrics(em_labels, em_preds, ud, rm_preds, od)
    print(f"Classifying exact match performance: precision {precision_em}, sensitivity {sensitivity_em}, f1 score {f1_em}")

    ## RELAX MATCH metrics
    precision_rm, sensitivity_rm, f1_rm, rm_cl_tp, rm_cl_fp, rm_cl_ud, rm_cl_od = relax_match_metrics(em_labels, em_preds, ud, rm_labels, rm_preds, od)
    print(f"Classifying relax match performance: precision {precision_rm}, sensitivity {sensitivity_rm}, f1 score {f1_rm}")


    ## Write csv so we can copy to results file
    metric_list = [sample, em_ex_tp, em_ex_fp, em_ex_ud, em_ex_od, em_cl_tp, em_cl_fp, em_cl_ud, em_cl_od, 
                                rm_ex_tp, rm_ex_fp, rm_ex_ud, rm_ex_od, rm_cl_tp, rm_cl_fp, rm_cl_ud, rm_cl_od] 
                
    import csv
    with open("no_sim_metric.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metric_list)