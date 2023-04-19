prompt = "Sex is between a"




from transformers import GPT2Tokenizer, OPTForCausalLM


from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("facebook/opt-6.7b")

with init_empty_weights():
  model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(model,no_split_module_classes=["OPTDecoderLayer"])

print(device_map)



#Load the model
model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")

#Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-6.7b")

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)
