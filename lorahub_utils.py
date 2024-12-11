from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import copy

def get_lora_module_names(num_modules=20):
    random.seed(42)
    return random.sample(LORA_MODULE_NAMES, num_modules)


# get a list of modules to be used in the composition
# modules = get_lora_modules()
# print("modules:", modules[:1])

def get_model(model_name="google/flan-t5-large", testing=True):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    if testing:
        inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return model, tokenizer


def get_lora_modules(lora_module_names, base_model):
    print(lora_module_names)
    random.seed(42)
    if lora_module_names == None:
        print("No LoRA modules sent")
    
    if base_model == None:
        print("No base model found.")

    lora_embedded_models = {}
    shape_matcher = None
    
    for lora_module in (lora_module_names):
        curr_model = PeftModel.from_pretrained(base_model, lora_module)
        lora_embedded_models[lora_module] = copy.deepcopy(get_peft_model_state_dict(curr_model))

        if shape_matcher is None:
            shape_matcher = lora_embedded_models[lora_module]
        try:
            for key in shape_matcher.keys():
                assert shape_matcher[key].shape == lora_embedded_models[lora_module][key].shape
        except:
            raise Exception(f'{lora_module} shape mismatch')
    
    return curr_model, lora_embedded_models
        

