# Jailbreak Testing Dashboard

The Jailbreak Testing Dashboard is a Streamlit-based tool for testing various prompt mutation techniques aimed at bypassing restrictions in language models. It supports both OpenAI and Ollama backends, allowing you to use your own OpenAI API key or fall back to a local LLM via Ollama. The project logs all attempts, provides detailed reporting, and features both single and batch testing modes.

## Features

- **Multiple Mutation Methods:**  
  Choose from methods such as rephrase, translation, backtranslation, obfuscation, token reversal, steganography, adversarial attacks, shuffling words, leet speak, and synonym substitution to modify prompts.

- **Flexible Backend Integration:**  
  Use your own OpenAI API key or switch to an Ollama-based local LLM backend.

- **Interactive Streamlit UI:**  
  A user-friendly dashboard allows you to enter prompts, select test modes, choose LLM models, and view detailed results and summary reports.

- **Logging and Reporting:**  
  All jailbreak attempts are logged in a JSON file and summarized in a report for easy analysis.

## Requirements

The project depends on the following Python libraries:

- [streamlit](https://streamlit.io/)
- [openai](https://github.com/openai/openai-python)
- [ollama](https://github.com/ollama/ollama) _(or similar local LLM integration)_

- [textattack](https://github.com/QData/TextAttack)
- [deep-translator](https://github.com/nidhaloff/deep-translator)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

A sample `requirements.txt` is provided:

```plaintext
streamlit
openai
ollama
textattack
deep-translator
python-dotenv
```

## Installation

### Clone the Repository

````bash
git clone https://github.com/yourusername/jailbreak-testing-dashboard.git
cd jailbreak-testing-dashboard

```bash
pip install -r requirements.txt



````
