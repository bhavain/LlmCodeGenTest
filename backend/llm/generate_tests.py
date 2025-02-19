from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
import os
from bs4 import BeautifulSoup
import json
from jsonschema import validate, ValidationError
import logging

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # LLM folder
PROMPT_FILE_PATH = os.path.join(BASE_DIR, "prompts/generate_tests.txt")

# Define a JSON schema for test cases
TEST_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"anyOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"}
                    ]},
                    "expected_output": {"anyOf": [
                        {"type": "string"},
                        {"type": "array"},
                        {"type": "object"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"}
                    ]}
                },
                "required": ["input", "expected_output"]
            }
        }
    },
    "required": ["test_cases"]
}


def extract_problem_details(problem_description):
    # Parse the HTML content
    soup = BeautifulSoup(problem_description, 'html.parser')
    
    # Extract Problem Name
    problem_name = soup.find('h1').text.strip() if soup.find('h1') else None
    
    # Extract Description (everything before the first <H2> tag)
    description_tag = soup.find('h2')
    description = ""
    if description_tag:
        for sibling in description_tag.find_previous_siblings():
            if sibling.name == 'p':
                description = sibling.text.strip() + "\n" + description
        description = description.strip()
    
    # Extract Model Input
    model_input = ""
    input_tag = soup.find('h2', string=lambda s: s and 'Input' in s)
    if input_tag:
        input_content = input_tag.find_next('p')
        model_input = input_content.text.strip() if input_content else ""
    
    # Extract Model Output
    model_output = ""
    output_tag = soup.find('h2', string=lambda s: s and 'Output' in s)
    if output_tag:
        output_content = output_tag.find_next('p')
        model_output = output_content.text.strip() if output_content else ""
    
    # Extract Sample Inputs and Outputs
    sample_inputs = []
    sample_outputs = []
    
    # Find all sample input/output tags
    sample_input_tags = soup.find_all('h2', string=lambda s: s and 'Sample Input' in s)
    for sample_input_tag in sample_input_tags:
        pre_tag = sample_input_tag.find_next('pre')
        if pre_tag:
            sample_inputs.append(pre_tag.text.strip())
    
    sample_output_tags = soup.find_all('h2', string=lambda s: s and ('Output for the Sample Input' in s or 'Sample Output' in s))
    for sample_output_tag in sample_output_tags:
        pre_tag = sample_output_tag.find_next('pre')
        if pre_tag:
            sample_outputs.append(pre_tag.text.strip())
    
    # Return all extracted details in a dictionary
    return {
        "problem_name": problem_name,
        "description": description,
        "model_input": model_input,
        "model_output": model_output,
        "sample_inputs": sample_inputs,
        "sample_outputs": sample_outputs
    }

def extract_input_output_json(test_cases_json):
    inputs = []
    outputs = []

    # Remove extra text and isolate JSON
    json_start = test_cases_json.find('{')
    json_end = test_cases_json.rfind('}')
    if json_start != -1 and json_end != -1:
        clean_json = test_cases_json[json_start:json_end + 1]
    else:
        logging.error("No valid JSON found.")
        return inputs, outputs

    # Parse and validate JSON
    try:
        # logging.info(f"Clean JSON: {clean_json}")
        test_cases = json.loads(clean_json)
        # logging.info(f"Test cases: {test_cases}")
        # test_cases = {'test_cases': [{'input': ['1819', '2003', '876', '2840', '1723', '1673', '3776', '2848', '1592', '922'], 'expected_output': ['3776', '2848', '2840']}, {'input': ['100', '200', '300', '400', '500', '600', '700', '800', '900', '900'], 'expected_output': ['900', '900', '800']}, {'input': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 'expected_output': ['10', '9', '8']}, {'input': ['10000', '9999', '9998', '9997', '9996', '9995', '9994', '9993', '9992', '9991'], 'expected_output': ['10000', '9999', '9998']}, {'input': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 'expected_output': ['9', '8', '7']}, {'input': ['20000', '10001', '5000', '3000', '2500', '1000', '500', '100', '50', '10'], 'expected_output': ['20000', '10001', '5000']}, {'input': [900, 900, 800, 700, 600, 500, 400, 300, 200, 100], 'expected_output': ['900', '900', '800']}, {'input': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'expected_output': ['100', '90', '80']}]}
        validate(instance=test_cases, schema=TEST_CASE_SCHEMA)  # Enforce schema

        for test_case in test_cases["test_cases"]:
            input_value = test_case["input"]
            expected_output = test_case["expected_output"]

            # logging.info(f"Input: {input_value}")
            # logging.info(f"Expected Output: {expected_output}")

            # Normalize input and output for saving
            normalized_input = normalize_data(input_value)
            normalized_output = normalize_data(expected_output)

            # logging.info(f"Normalized Input: {normalized_input}")
            # logging.info(f"Normalized Output: {normalized_output}")

            inputs.append(normalized_input)
            outputs.append(normalized_output)

    except ValidationError as e:
        logging.error(f"JSON does not match schema: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")

    return inputs, outputs

def normalize_data(value):
    """
    Convert the input/output value to a normalized string representation for saving.
    Handles different types (e.g., array, object, string).
    """
    if isinstance(value, list):
        return " ".join(map(str, value))  # Join list elements with a space
    if isinstance(value, dict):
        return json.dumps(value)  # Convert to JSON string
    elif isinstance(value, (int, float, bool, type(None))):
        return str(value)  # Convert to string
    elif isinstance(value, str):
        return value.strip()
    else:
        return str(value)  # Fallback for unknown types

def generate_test_cases(problem_description):
    """Uses an LLM to generate structured test cases."""
    print("Generating test cases...")
    llm = ChatOllama(model="llama3.1")

    # Extract input/output format and constraints from the problem description
    problem_details = extract_problem_details(problem_description)

    with open(PROMPT_FILE_PATH, 'r') as file:
        prompt = file.read()

    filled_prompt = prompt.format(
        problem_name=problem_details["problem_name"],
        description=problem_details["description"],
        model_input=problem_details["model_input"],
        model_output=problem_details["model_output"],
        sample_inputs=problem_details["sample_inputs"],
        sample_outputs=problem_details["sample_outputs"],
        num_cases=20
    )

    response = llm.invoke(filled_prompt)

    # Extract and validate JSON from the response
    parsed_response = response.content
    inputs, outputs = extract_input_output_json(parsed_response)
    test_cases = [{"input": inp, "expected_output": out} for inp, out in zip(inputs, outputs)]

    return test_cases  # Expecting structured JSON
