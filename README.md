# LLM (Large Language Model) based pipeline for Review of Systems Entity Recognition from Clincal Notes

## Prerequisite
Install dependencies in requirements.txt: !pip install -r ./requirements.txt
Install ollama from https://ollama.com/download

### Model configuration files  
For problem detection, negation detection & body system classification, model files are ros_classify and ros_extract. Create models by running the following script:

#!/bin/bash
ollama create ros_extract -f ./ros_extract
ollama create ros_classify -f ./ros_classify

## Pipeline
- label_processing.py -> pipeline.py -> evaluation.py (with attribution vs no attribution: indicated in filename)
- pipeline.py is intended to run for each note (with --filename argument) - to carefully evaluate the LLM's behavior
- evaluation.ipynb notebook is for generic and some extra evaluation purposes

## Label: stored in input folder:
- label_classify.json: label for body system classification
- label_extract.json: label for negation detection
- label.json: label for problem detection

## Output: stored in folders with "Output_" prefix
folder with "no_sim": without attribution
- output_classify.json: entities extracted for body system classification
- output_extract.json: entities extracted for negation detection
- output.json: entities extracted for problem detection

folder with only model name: with attribution
- output_classify.json: entities extracted for body system classification after attribution
- output_extract.json: entities extracted for negation detection after attribution
- output_original.json: entities extracted for problem detection, without attribution
- output.json: entities extracted for problem detection after attribution

## Dataset
Medical transcription samples and reports used as our training and benchmark dataset are available on https://www.mtsamples.com

If you have questions, feel free to contact us!