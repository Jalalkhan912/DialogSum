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

example_indices_full = [40]
example_indices_full_few_shot = [40, 80, 120]
example_index_to_summarize = 0
dash_line = '-'.join('' for x in range(100))

def zero_shot(example_index_to_summarize):
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    summary = dataset['test'][example_index_to_summarize]['summary']

    prompt = f"""
Dialogue:

{dialogue}

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

    return summary, output

def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    dialogue = dataset['test'][example_index_to_summarize]['dialogue']

    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""

    return prompt

def one_shot(example_index_to_summarize):
  summary = dataset['test'][example_index_to_summarize]['summary']

  inputs = tokenizer(one_shot_prompt, return_tensors='pt')
  output = tokenizer.decode(
      model.generate(
          inputs["input_ids"],
          max_new_tokens=50,
      )[0],
      skip_special_tokens=True
  )
  return output

def few_shot(example_index_to_summarize):
  summary = dataset['test'][example_index_to_summarize]['summary']

  inputs = tokenizer(few_shot_prompt, return_tensors='pt')
  output = tokenizer.decode(
      model.generate(
          inputs["input_ids"],
          max_new_tokens=50,
      )[0],
      skip_special_tokens=True
  )
  return output

st.title("FLAN-T5(Base) Prompt Engineered: Zero-shot, Single-shot, and Few-shot")
import pandas as pd

data = pd.read_csv('test_data_sample.csv')

st.write(data)
example_index_to_summarize = st.number_input(
    label="Enter Index to Summarize",
    min_value=0,
    max_value=100,
    value=0
)  
if st.button("Run"):  
    summary, zero_shot_output = zero_shot(example_index_to_summarize)
    one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
    one_shot_output = one_shot(example_index_to_summarize)
    few_shot_prompt = make_prompt(example_indices_full_few_shot, example_index_to_summarize)
    few_shot_output = few_shot(example_index_to_summarize)
if st.button("Results"):
    st.header("Comparizion of Outputs")
    st.write(f"**BASELINE HUMAN SUMMARY:**\n{summary}\n")
    st.write(f"**Zero-shot Output:**\n{zero_shot_output}\n")
    st.write(f"**Single-shot Output:**\n{one_shot_output}\n")
    st.write(f"**Few-shot Output:**\n{few_shot_output}\n")
