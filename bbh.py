import os
import json
import random
from lorahub_learning import *
import zipfile

def evaluate_bbh(folder='data_bbh'):
    # if not os.path.exists("data_bbh"):
        # os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # os.system("unzip data_bbh.zip")
    
    # with zipfile.ZipFile('data_bbh.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./')

    sub_dirs = os.listdir(folder)
    sub_dirs = ['salient_translation_error_detection', 'logical_deduction_five_objects', 'movie_recommendation', 'geometric_shapes', 'hyperbaton', 'reasoning_about_colored_objects', 'navigate', 'object_counting']

    for sub_dir in sub_dirs:
        print("Task: ", sub_dir.replace('_', ' ').title())
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        
        num_loras = [3, 5, 7, 10, 15, 20, 35]
        for n in num_loras:
            final_model, tokenizer = adapt_lora(0.25, 100, n, example_inputs, examples_outputs, task_inputs, task_outputs)

evaluate_bbh()
