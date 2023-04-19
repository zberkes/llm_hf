import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

prompt = "Why did the chicken cross the road?"
max_len = 100

# init
torch.cuda.set_per_process_memory_fraction(0.90, 0)

checkpoint = "cerebras/Cerebras-GPT-13B"
offload_folder = "offload"


# load
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto", offload_folder=offload_folder)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", offload_folder=offload_folder)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.float16)

# Perform inference
start_inference = time.time()
response = pipe(prompt, max_length=max_len, do_sample=False, no_repeat_ngram_size=2)[0]['generated_text']
print(response)

# done
end_time = time.time()


load_time = start_inference - start_time
inference_time = end_time - start_inference
total_time = end_time - start_time

mm1, ss1 = divmod(total_time, 60)
mm2, ss2 = divmod(load_time, 60)
mm3, ss3 = divmod(inference_time, 60)

print()
print("\nElapsed time: {:02d}:{:02d} (load: {:02d}:{:02d}, inference: {:02d}:{:02d})".format(
    int(mm1), int(ss1),
    int(mm2), int(ss2),
    int(mm3), int(mm3))
)

text_len = len(response)
speed = float(text_len) / inference_time
print("{:d} chars, {:.1f} chars/s".format(text_len, speed))
