import copy
from functools import partial
from typing import List
from learning_examples import get_examples_for_learning, get_examples_for_inference
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from lorahub_utils import get_lora_module_names, get_model, get_lora_modules
import nevergrad
import torch
import random
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

def tokenize(examples, tokenizer):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def create_dataset(inputs: List[str], outputs: List[str], combined_ds, tokenizer):
    if combined_ds == None:
        if len(outputs) == 0:
            outputs = [""] * len(inputs)

        dataset = []
        for i in range(len(inputs)):
            dataset.append({"input": inputs[i], "output": outputs[i]})
    else:
        dataset=combined_ds

    dataset = Dataset.from_list(dataset)
    tokenized_dataset = dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True)
    return tokenized_dataset

def compose_lora_modules(base_model, lora_models, weights):
    composed_lora = {}
    keys = lora_models[list(lora_models.keys())[0]].keys()
    for i, lora_model_name in enumerate(lora_models):
        lora_model = lora_models[lora_model_name]
        for key in keys:
            if key not in composed_lora:
                composed_lora[key] = weights.value[i] * lora_model[key]
            else:
                composed_lora[key] += weights.value[i] * lora_model[key]

    # print("Composed LoRA")
    # print(composed_lora)
    set_peft_model_state_dict(base_model, composed_lora)
    # print("Adding")
    # for name, param in base_model.named_parameters():
    #     if name in composed_lora:
    #         print("Matched: ", name, base_model.state_dict()[name].shape)
    #         base_model.state_dict()[name] += composed_lora[name]

    # print("Composed Model: ")

    # for name, param in base_model.named_parameters():
    #     print(f"{name}: {param.shape}")
    #     print(base_model.state_dict()[name])
    #     break
    return base_model

def loss(optimal_weights, model, dataset, batch_size, reg_factor=0.05):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    batch_size = len(dataset) if batch_size is None else min(len(dataset), batch_size)
    dataloader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    train_loss = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()

    reg_loss = reg_factor * (sum([abs(x) for x in optimal_weights]) / len(optimal_weights))
    return float(loss) / len(dataset) + reg_loss

def adapt_lora(init_value = 0.1, budget = 100, num_loras=20, learning_sample_inputs=None, learning_sample_outputs=None, testing_sample_inputs=None, testing_sample_outputs=None):
    print("Init: ", init_value) #, "Iters: ", budget)
    model, tokenizer = get_model(testing=False)

    lora_module_names = get_lora_module_names(num_modules=num_loras)
    num_loras = len(lora_module_names)
    if learning_sample_inputs == None:
        learning_samples = get_examples_for_learning()
        dataset = create_dataset(inputs=None, outputs=None, combined_ds=learning_samples, tokenizer=tokenizer)
    else:
        dataset = create_dataset(learning_sample_inputs, learning_sample_outputs, combined_ds=None, tokenizer=tokenizer)
    # print(type(lora_models[lora_module_names[0]]))

    # for kv in model.named_parameters():
    #     print(kv)
    #     break

    if testing_sample_inputs == None:
        testing_samples = get_examples_for_inference()
        test_dataset = create_dataset(None, None, testing_samples, tokenizer)
    else:
        test_dataset = create_dataset(testing_sample_inputs, testing_sample_outputs, combined_ds=None, tokenizer=tokenizer)

    # predictions = inference(model, tokenizer, test_dataset)
    # taks_accuracy =  accuracy(predictions, test_dataset['output'])
    # print("Original accuracy:", taks_accuracy)

    model, lora_models = get_lora_modules(base_model=model, lora_module_names=lora_module_names)


    instrum = nevergrad.p.Array(
        init=[init_value] * num_loras,
        upper=[50] * num_loras,
        lower=[-50] * num_loras,
    )
    batch_size=4
    print(instrum.value)
    optimizer = nevergrad.optimizers.NGOpt(parametrization=instrum, budget=budget)
    lora_weights = optimizer.minimize(partial(loss, model=model, dataset=dataset, batch_size=batch_size))

    final_model = compose_lora_modules(model, lora_models, lora_weights)
    print(lora_weights.value)
    print("Running inference")
    predictions = inference(final_model, tokenizer, test_dataset)
    taks_accuracy =  accuracy(predictions, test_dataset['output'])
    print("Accuracy: ", taks_accuracy)

    # for kv in final_model.named_parameters():
    #     print(kv)
    #     break

    return final_model, tokenizer

def inference(model, tokenizer, test_dataset, batch_size=1):
    batch_size = len(test_dataset) if batch_size is None else min(len(test_dataset), batch_size)
    dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # batch.to(device)
            with torch.no_grad():
                outputs = model.generate(**batch, max_new_tokens=1024)
                outputs = (tokenizer.batch_decode(outputs, skip_special_tokens=True))
                all_outputs += outputs

    return all_outputs

def accuracy(predictions, true_outputs):
    if len(predictions) != len(true_outputs):
        raise ValueError("The number of predictions and true outputs must be the same.")
    
    correct_predictions = sum([1 for p, t in zip(predictions, true_outputs) if p.strip().lower().replace(".", "") == t.strip().lower().replace(".", "")])
    
    accuracy = (correct_predictions / len(predictions)) * 100
    return accuracy

final_model, tokenizer = adapt_lora(init_value=5, budget=1)















    
