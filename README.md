# DialogSum: Dialogue Summarization with FLAN-T5

## Overview
DialogSum is an AI-powered dialogue summarization tool that uses Google’s FLAN-T5 base model to produce concise summaries of multi-turn conversations. It implements **Zero-shot**, **One-shot**, and **Few-shot** summarization techniques via prompt engineering. The project provides an interactive web UI built with Streamlit, allowing users to input dialogues and compare different summarization modes side-by-side.

## Features
- **Zero-shot Summarization**: Generate summaries without any example context.
- **One-shot Summarization**: Utilize a single example for improved contextual summarization.
- **Few-shot Summarization**: Enhance summarization by providing multiple examples.
- **Streamlit Interface**: An easy-to-use web UI that displays outputs from each summarization mode for comparison.

## Model Details
- **Model Used**: Google’s **FLAN-T5-base**—a fine-tuned transformer model optimized for text generation tasks.
- **Dataset**: The **DialogSum** dataset from Hugging Face, containing real-world dialogues with summaries.
- **Tokenizer**: Tokenization is managed by the FLAN-T5-specific fast tokenizer.

## How It Works
1. **User Input**: Users enter a multi-turn conversation via the Streamlit interface.
2. **Summarization Methods**:
   - **Zero-shot**: Generates a summary without any example reference.
   - **One-shot**: Uses one dialogue-summary pair to guide the output.
   - **Few-shot**: Incorporates several examples to refine summarization quality.
3. **Display**: Summaries from each mode are presented side-by-side for easy comparison.

## Code Structure
- **Prompt Engineering**: Custom prompts format dialogue input for each mode.
- **Model Generation**: Configurations like `max_new_tokens` and `temperature` control output.
- **Example Handling**: Selects example dialogues from the DialogSum dataset for one-shot and few-shot modes.

## Use Cases
- **Customer Support**: Summarize customer-agent conversations for insights.
- **Meeting Minutes**: Extract key points from meeting transcripts.
- **Chatbot Analysis**: Summarize and evaluate chatbot interactions.
- **Education**: Help students understand conversation summaries in various contexts.

## Future Improvements
- **Temperature Control in UI**: Allow users to adjust temperature for creative summarization.
- **Model Fine-tuning**: Fine-tune FLAN-T5 on other conversation datasets for enhanced domain-specific performance.
- **Multilingual Summarization**: Extend support for dialogues in other languages.
- **Real-time Streaming**: Enable summarization from live conversations.

