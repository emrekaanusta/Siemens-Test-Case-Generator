# PDF‑to‑JSON Test Suite Generator
A lightweight Python tool that converts Siemens SIMATIC specification PDFs into a validated JSON test suite—complete with negative‑test equivalence classes—and provides an interactive web UI for execution and auto‑refinement.

# Features

- One‑click JSON generation from any Siemens specification PDF.

- Schema validation against the official Test Configuration Schema v1.4.

- Negative testing via 15 auto‑generated equivalence classes.

- Interactive Gradio UI for upload, inspection, execution, and auto‑refinement.

- Extensible—backend parameter k already supports any number of equivalence classes.



# Quick Start

Tested on Python 3.11 (Windows).  Linux/macOS users can adapt the activate command accordingly.

# 1. Clone the repository
$ git clone git@sabanci.edu:ekusta/pdf2json.git
$ cd pdf2json

# 2. Create and activate a virtual environment
$ py -m venv .venv
$ .venv\Scripts\activate  # Windows

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Add your OpenAI key
$ echo OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx > key.env

# 5. Launch the tool
$ python generate_test.py
# Gradio interface opens at http://127.0.0.1:7860


# Usage Tips

- Generating JSON

- Upload a Siemens specification PDF.

- Choose a model (chatgpt:3.5, gpt-4o, or o1).

- Click Generate JSON.



# Validating & Refining

- Select an individual test case from the dropdown or choose All Test Cases.

- Click Run Selected Test Case to execute via test_case_checker.

- If the result deviates from expected_return, click Validate & Auto‑Refine to let GPT‑4‑o correct the payload.


# Troubleshooting

Issue                                                                             Fix

Valid Configuration Schema box is empty                                           Hit Generate JSON again; rare LLM formatting glitch.

Feedback ignored when All Test Cases is shown                                     Re‑select All Test Cases from the dropdown (click it).

Model o1 returns nothing                                                          Try a different API key or switch to chatgpt:3.5.  Report results.


# License

This project is released under the MIT License.  Siemens retains ownership of all JSON artefacts generated from proprietary documents.






