import os
import json
import re
import openai
import gradio as gr
import subprocess
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import json5
import json
import json5
import yaml
import gradio as gr

# Load OpenAI API key
load_dotenv("key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Loaded API Key:", openai.api_key)

#######################################
# Utility: Clean JSON output by extracting block & removing trailing commas
#######################################
def clean_json_output(text: str) -> str:
    import re, json

    # Locate first “{” or “[”
    obj_start = text.find('{')
    arr_start = text.find('[')
    if obj_start == -1 and arr_start == -1:
        return text                     # no JSON‑looking chunk found

    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        # Looks like an array
        start, end = arr_start, text.rfind(']')
    else:
        # Looks like an object
        start, end = obj_start, text.rfind('}')

    if end == -1 or end <= start:
        return text                     # malformed

    block = text[start:end + 1]

    # Optional tidy‑ups (same as before)
    block = re.sub(r',\s*,+', ',', block)              # collapse duplicate commas
    block = re.sub(r',\s*([}\]])', r'\1', block)       # strip trailing commas

    # Last sanity check
    try:
        json.loads(block)
    except Exception:
        pass

    return block

#######################################
# 1) PDF Text Extraction
#######################################
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

#######################################
# 2) LLM Query
#######################################
def query_llm_pdf(text_content, model_name, system_prompt):
    user_prompt = f"System instructions:\n{system_prompt}\n\nPDF Content:\n{text_content}\n"
    if model_name == 'chatgpt:3.5':
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role':'system','content':'You are a helpful assistant.'},
                {'role':'user','content':user_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
    else:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {'role':'system','content':'You are a helpful assistant.'},
                {'role':'user','content':user_prompt}
            ],
            max_completion_tokens=3000
        )
    return (response.choices[0].message.content or "").strip()

#######################################
# 3) Prompts
#######################################
pdf_prompt = '''
I will provide a technical manual that describes the configuration and commissioning of a system. My goal is to validate JSON configuration files against the exact structure provided in the manual. I need you to:

Extract all required and optional fields for the JSON configuration files from the manual (including field names, expected data types, and constraints).
Ensure that the valid JSON configuration file matches the factory settings as described in the document, without any modification or assumption.
List all valid values for each field, including any explicitly stated constraints (e.g., allowed interface names, numerical limits, and boolean requirements).
Define 15 invalid JSON equivalence classes, ensuring that:
  Each class targets a specific failure scenario (e.g., missing required fields, wrong data types, incorrect values).
  Each test case within a class preserves the basic JSON structure while introducing only one fault at a time to isolate validation errors.
The test cases do not remove all required fields at once but instead focus on granular errors (e.g., nvram_usage: "true" instead of true rather than omitting it altogether).
Generate a JSON file containing:
  A list of 15 equivalence classes, each with:
    class_name: A unique name describing the failure scenario.
    description: A brief explanation of what is being tested.
    test_cases: A list of cases, each including:
      json_name: The test case filename that is descriptive.
      json_description: A short explanation of the issue.
      expected_return: The expected output from the executable when this test case is run.
      json: The invalid JSON content.
The JSON should not contain random values; all values must be based only on the document.
Important Considerations:

Do not introduce assumptions about missing values—follow the document exactly.
Keep the correct default configuration file unaltered in a separate section before listing invalid test cases.
Ensure all test cases remain structurally correct apart from their intended errors.

**IMPORTANT:** Your final answer **must** be **valid JSON only**, with exactly two top‑level keys and **no** stray commas, comments, or prose:

json
{
  "default_configuration": { /* full default settings */ },
  "invalid_equivalence_classes": [ /* array of test‑classes */ ]
}
'''

#######################################
# 4) Generate JSON
#######################################
def process_pdf(pdf_file, model_name):
    text = extract_pdf_text(pdf_file)
    prompt = pdf_prompt + "\nYou MUST output raw JSON only."
    raw = query_llm_pdf(text, model_name, prompt)
    return clean_json_output(raw)

#######################################
# 5) Split valid vs invalid + dropdown
#######################################
def update_test_case_options(full_json_str):
    try:
        data = json.loads(full_json_str)
        choices = []
        for cls in data.get('invalid_equivalence_classes', []):
            for tc in cls.get('test_cases', []):
                idx = len(choices)
                name = tc.get('json_name', '')
                exp  = tc.get('expected_return', '')
                label = f"{idx}: {name}" + (f" (expect → {exp})" if exp else "")
                choices.append(label)
        if choices:
            choices.insert(0, 'All: All Test Cases')
        else:
            choices = ['No test cases found']
        return gr.update(choices=choices, value=choices[0])
    except:
        return gr.update(choices=['No test cases found'], value='No test cases found')

def process_pdf_and_update_dropdown(pdf_file, model_name):
    import json, json5, yaml, gradio as gr

    # 1) Generate & clean the raw JSON wrapper
    raw     = process_pdf(pdf_file, model_name)
    cleaned = clean_json_output(raw)

    # 2) Try to parse with strict JSON → JSON5 → YAML
    parse_error = None
    data = None
    try:
        data = json.loads(cleaned)
    except Exception as e1:
        parse_error = e1
        try:
            data = json5.loads(cleaned)
        except Exception as e2:
            parse_error = e2
            try:
                data = yaml.safe_load(cleaned)
            except Exception as e3:
                parse_error = e3
                data = None

    # 3) If parsing failed completely, show raw cleaned text and disable dropdown
    if data is None or not isinstance(data, dict):
        # valid_schema might still parse defaults in some cases, but we'll leave it blank
        valid_str = ""
        # show the cleaned block so you can debug it manually
        dropdown = gr.update(choices=["No test cases found"], value="No test cases found")
        return valid_str, cleaned, dropdown

    # 4) Otherwise extract the two pieces
    valid_default = data.get("default_configuration", {})
    invalid_list  = data.get("invalid_equivalence_classes", [])

    # 5) Pretty-print for the UI
    valid_str = json.dumps(valid_default, indent=2)
    cases_str = json.dumps(invalid_list, indent=2)

    # 6) Rebuild the dropdown from the invalid list
    wrapper  = json.dumps({"invalid_equivalence_classes": invalid_list})
    dropdown = update_test_case_options(wrapper)

    return valid_str, cases_str, dropdown

#######################################
# 6) Refine JSON
#######################################


#######################################
# 7) Validate & Auto-Refine
#######################################
def validate_and_refine_test_case(tc, exe, model, flag, force, verb):
    attempt=0; expected=str(tc.get('expected_return','')).strip().lower()
    while attempt<3:
        with open('temp_tc.json','w') as f: json.dump(tc['json'],f)
        cmd=[exe,flag,'temp_tc.json']
        if force: cmd.append('--force-reboot')
        if verb:  cmd.append('--verbose')
        try:
            res=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            actual=res.stdout.strip().lower()
        except:
            return tc,f"Error on run"
        if actual==expected:
            return tc,f"✅ Match on attempt {attempt+1}"
        prompt=f"""
Given test case: {json.dumps(tc)}
It returned '{actual}', expected '{expected}'.
Return ONLY updated JSON.
"""
        raw=query_llm_pdf('',model,prompt)
        clean=clean_json_output(raw)
        try:
            new_tc=json.loads(clean)
            tc.clear(); tc.update(new_tc)
        except:
            return tc,f"❌ Invalid JSON refine {attempt+1}"
        attempt+=1
    return tc,f"⚠️ Max attempts reached; last {actual}"

def run_validation_cycle(exe, cases_json, selected, model, flag, force, verbose):
    """
    Runs a single selected test case through the checker and, if needed,
    auto‑refines its JSON payload. Always returns three values:
      1) updated cases_json (string)
      2) status message (string)
      3) dropdown update for the test case selector
    """
    import json

    # 1) Load the current list of test‑classes
    try:
        lst = json.loads(cases_json)
    except json.JSONDecodeError:
        # If the JSON is malformed, return it unchanged
        return cases_json, "❌ Unable to parse test cases JSON", gr.update(
            choices=["No test cases found"], value="No test cases found"
        )

    data = {"invalid_equivalence_classes": lst}

    # 2) Flatten all test cases into a list
    flat, idx_map = [], []
    for ci, cls in enumerate(lst):
        for ti, tc in enumerate(cls.get("test_cases", [])):
            flat.append(tc)
            idx_map.append((ci, ti))

    # 3) Early‑exit if user selected "All" or nothing valid
    if selected.lower().startswith("all"):
        wrapper   = json.dumps({"invalid_equivalence_classes": lst})
        dd_update = update_test_case_options(wrapper)
        return cases_json, "⚠️ Please select a single test case to validate", dd_update

    # 4) Parse the selected index
    try:
        idx = int(selected.split(":")[0])
    except:
        wrapper   = json.dumps({"invalid_equivalence_classes": lst})
        dd_update = update_test_case_options(wrapper)
        return cases_json, "❌ Invalid selection", dd_update

    if idx < 0 or idx >= len(flat):
        wrapper   = json.dumps({"invalid_equivalence_classes": lst})
        dd_update = update_test_case_options(wrapper)
        return cases_json, "❌ Index out of range", dd_update

    # 5) Validate & potentially auto‑refine that one test case
    tc, msg = validate_and_refine_test_case(flat[idx], exe, model, flag, force, verbose)

    # 6) Patch it back into our data structure
    ci, ti = idx_map[idx]
    data["invalid_equivalence_classes"][ci]["test_cases"][ti] = tc

    # 7) Serialize the updated list and rebuild the dropdown
    new_list_str = json.dumps(data["invalid_equivalence_classes"], indent=2)
    wrapper      = json.dumps({"invalid_equivalence_classes": data["invalid_equivalence_classes"]})
    dd_update    = update_test_case_options(wrapper)

    return new_list_str, msg, dd_update
def refine_json(invalid_str, feedback, model_name, selected_option):
    """
    - If selected_option is NOT a single-number index (e.g. "2: foo.json"),
      we treat it as a global refine: we send the entire array + feedback.
    - Otherwise we only refine the one test case as before.
    Returns the updated JSON-ARRAY string of equivalence classes.
    """
    import json, re

    # 1) load array
    try:
        arr = json.loads(invalid_str)
    except Exception:
        return invalid_str

    # 2) decide global vs per-CI
    is_global = not re.match(r'^\d+:', selected_option.strip())

    if is_global:
        # —— GLOBAL REFINE ——
        prompt = f"""
Here is the full list of invalid_equivalence_classes:
{json.dumps(arr, indent=2)}

User feedback: {feedback}

Please apply that feedback and return ONLY the updated JSON **array** of equivalence‑classes.
"""
        raw   = query_llm_pdf('', model_name, prompt)
        clean = clean_json_output(raw)
        try:
            new_arr = json.loads(clean)
            return json.dumps(new_arr, indent=2)
        except Exception:
            # if the LLM blew up, fall back
            return invalid_str

    # —— PER-TEST-CASE REFINE ——  
    # flatten out test cases
    data    = {'invalid_equivalence_classes': arr}
    tcs, idx_map = [], []
    for ci, cls in enumerate(arr):
        for ti, tc in enumerate(cls.get('test_cases', [])):
            tcs.append(tc)
            idx_map.append((ci, ti))

    # pick the single index
    m = re.match(r'^(\d+):', selected_option.strip())
    if not m:
        return invalid_str
    idx = int(m.group(1))
    if idx < 0 or idx >= len(tcs):
        return invalid_str

    # prompt just that one test-case
    tc   = tcs[idx]
    orig = json.dumps(tc, indent=2)
    prompt = f"""
You are given this single test‑case object:
{orig}

User feedback: {feedback}

Return ONLY the updated test‑case JSON (including its full 'json' payload).
"""
    raw   = query_llm_pdf('', model_name, prompt)
    clean = clean_json_output(raw)
    try:
        parsed = json.loads(clean)
    except:
        return invalid_str

    ci, ti = idx_map[idx]
    existing = data['invalid_equivalence_classes'][ci]['test_cases'][ti]

    # if they returned a full test‑case dict, swap it in; else merge inner payload
    if isinstance(parsed, dict) and 'json' in parsed:
        data['invalid_equivalence_classes'][ci]['test_cases'][ti] = parsed
    else:
        existing['json'] = parsed
        data['invalid_equivalence_classes'][ci]['test_cases'][ti] = existing

    return json.dumps(data['invalid_equivalence_classes'], indent=2)
#######################################
# 8) Run Test Case
#######################################
def run_test_case(exe, cases_json, selected, flag, force, verbose):
    # parse the current (possibly refined) list
    lst = json.loads(cases_json)
    all_tcs = [tc for cls in lst for tc in cls.get('test_cases', [])]

    if selected.lower().startswith("all"):
        sel_tcs = all_tcs
    else:
        idx = int(selected.split(":")[0])
        sel_tcs = [all_tcs[idx]]

    outputs = []
    for tc in sel_tcs:
        with open("temp_tc.json", "w") as f:
            json.dump(tc["json"], f)
        cmd = [exe, flag, "temp_tc.json"]
        if force:   cmd.append("--force-reboot")
        if verbose: cmd.append("--verbose")
        res = subprocess.run(cmd, capture_output=True, text=True)
        outputs.append(res.stdout.strip())
    return "\n---\n".join(outputs)




def on_refine(cases_json, feedback, model_name, selected):
    # 1) run your refine_json to produce a NEW list of test‑classes
    refined_list_str = refine_json(cases_json, feedback, model_name, selected)

    # 2) regenerate the dropdown from the refined list
    dropdown = update_test_case_options(
        json.dumps({"invalid_equivalence_classes": json.loads(refined_list_str)})
    )

    # 3) return the **same** cases‑textbox + dropdown
    return refined_list_str, dropdown
    



#######################################
# 9) Build Gradio Interface
#######################################
available_models=['chatgpt:3.5','gpt-4o','o1']
default_model='chatgpt:3.5'

with gr.Blocks() as demo:
    gr.Markdown("## PDF → JSON + Feedback + Execution")

    with gr.Row():
        with gr.Column():
            pdf_file    = gr.File(label="Upload PDF", file_types=[".pdf"])
            model_sel   = gr.Radio(["chatgpt:3.5","gpt-4o","o1"], value="chatgpt:3.5")
            gen_btn     = gr.Button("Generate JSON")


            test_case_dd = gr.Dropdown(
                label="Select Test Case",
                choices=["No test cases found"],
                value="", 
                allow_custom_value=True
)

            feedback_in  = gr.Textbox(label="Feedback/Corrections", lines=5)
            refine_btn   = gr.Button("Refine Selected Test Case")


        with gr.Column():

            valid_schema = gr.Textbox(label="Valid Configuration Schema", lines=10)
            test_cases   = gr.Textbox(label="Test Cases JSON", lines=15)
            validation_status = gr.Textbox(label="Validation Status", lines=3)

    with gr.Row():
        exe_path    = gr.Textbox(label="Executable Path", value="dist/test_case_checker.exe")
        resource_f  = gr.Radio(["--resource-config","-r"], value="--resource-config", label="Resource Flag")
        force_rb    = gr.Checkbox(label="--force-reboot")
        verbose_cb  = gr.Checkbox(label="--verbose")


    run_output = gr.Textbox(label="Execution Output", lines=10)

    with gr.Row():

        run_btn     = gr.Button("Run Selected Test Case")
        validate_btn= gr.Button("Validate & Auto‑Refine")
    
    # ——— Wiring ———
    gen_btn.click(
        fn=process_pdf_and_update_dropdown,
        inputs=[pdf_file, model_sel],
        outputs=[valid_schema, test_cases, test_case_dd]
    )

    refine_btn.click(
        fn=on_refine,
        inputs=[test_cases, feedback_in, model_sel, test_case_dd],
        outputs=[test_cases, test_case_dd]
    )

    run_btn.click(
        fn=run_test_case,
        inputs=[exe_path, test_cases, test_case_dd, resource_f, force_rb, verbose_cb],
        outputs=[run_output]
    )

    validate_btn.click(
        fn=run_validation_cycle,
        inputs=[exe_path, test_cases, test_case_dd, model_sel, resource_f, force_rb, verbose_cb],
        outputs=[test_cases, validation_status, test_case_dd]
    )

demo.launch(share=True)