# load important libraries
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import streamlit as st

# load the dialog summarization dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

# load the google FLAN-T5 base model
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# load the specific tokenizer for above model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# initialize variables
example_indices_full = [40]
example_indices_full_few_shot = [40, 80, 120, 200, 220]
dash_line = '-'.join('' for x in range(100))

# zero_shot inference
def zero_shot(my_example):
    prompt = f"""
Dialogue:

{my_example}

What was going on?
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    return output

# this prompt template will be used
def my_prompt(example_indices, my_example):
    prompt = ''
    for index in example_indices:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    prompt += f"""
Dialogue:

{my_example}

What was going on?
"""

    return prompt


# this is for one_shot
def one_shot(example_indices_full,my_example):
  inputs = tokenizer(my_prompt(example_indices_full,my_example), return_tensors='pt')
  output = tokenizer.decode(
      model.generate(
          inputs["input_ids"],
          max_new_tokens=50,
      )[0],
      skip_special_tokens=True
  )
  return output

# few_shot
def few_shot(example_indices_full_few_shot,my_example):
  inputs = tokenizer(my_prompt(example_indices_full_few_shot,my_example), return_tensors='pt')
  output = tokenizer.decode(
      model.generate(
          inputs["input_ids"],
          max_new_tokens=50,
      )[0],
      skip_special_tokens=True
  )
  return output

st.title("FLAN-T5(Base) Prompt Engineered: Zero-shot, Single-shot, and Few-shot")

my_example = st.text_area("Enter dialogues to summarize", value="#Maaz#: Jalal how are you?#Jalal#:  I am good thank you.#Maaz#: Are you going to school tomorrow.#Jalal#: No bro i am not going to school tomorrow.#Maaz#: why? #Jalal#: I am working on a project, are you want to work with me on my project?#Maaz#: sorry, i have to go to school.")

if st.button("Run"):  
    zero_shot_output = zero_shot(my_example)
    one_shot_output = one_shot(example_indices_full,my_example)
    few_shot_output = few_shot(example_indices_full_few_shot,my_example)
    st.header("Comparizion of Outputs")
    st.write(f"**Zero-shot Output:**\n{zero_shot_output}\n")
    st.write(f"**Single-shot Output:**\n{one_shot_output}\n")
    st.write(f"**Few-shot Output:**\n{few_shot_output}\n")
