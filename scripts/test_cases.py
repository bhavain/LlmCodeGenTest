import os
import ollama
import json

import logging
from jsonschema import validate, ValidationError
from bs4 import BeautifulSoup
import re

from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)

from operator import itemgetter
from typing import List

from groq import Groq

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# Configure logging
logging.basicConfig(
    filename="automation.log",  # Log file name
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

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

# Initialize Ollama
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0.2,
#     # other params...
# )

# # Prompt template for generating test cases step-by-step
# prompt = PromptTemplate(
#     input_variables=["history", "problem_description", "step"],
#     template="""
#     You are a JSON generation assistant specializing in creating programming test cases.

#     Problem Description:
#     {problem_description}

#     Task: {step}

#     Current conversation: {history}

#     Ensure the output follows this strict JSON format:
#     {{
#         "test_cases": [
#             {{
#                 "input": "<formatted input>",
#                 "expected_output": "<formatted output>"
#             }},
#             ...
#         ]
#     }}

#     Guidelines:
#     - Validate and ensure inputs match the Input Format.
#     - Compute outputs logically based on the Output Format.
#     - Do not include explanations or additional text.
#     """
# )

# # Initialize memory
# memory = ConversationBufferMemory()

# # Conversation chain for context preservation
# conversation_chain = ConversationChain(llm=llm, prompt=prompt, memory = memory)

# chain = prompt | llm

# conversation_chain = RunnableWithMessageHistory(
#     chain,
#     # Uses the get_by_session_id function defined in the example
#     # above.
#     get_by_session_id,
#     custom_input_type=List[str],
#     input_messages_key={"problem_description", "input_format", "output_format", "step"},
#     history_messages_key="history",
# )


def strip_html_tags(html):
    """
    Strips HTML tags from a given string.

    Args:
    html (str): String containing HTML content.

    Returns:
    str: Text content without HTML tags.
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def extract_format_from_description(problem_description):
    """
    Extract the input and output format from a structured problem description.
    Returns the input format, output format, and constraints as strings.
    """
    input_format = ""
    output_format = ""
    constraints = ""

    if "Input" in problem_description:
        input_start = problem_description.find("Input")
        output_start = problem_description.find("Output")
        sample_input_start = problem_description.find("Sample Input")
        # constraints_start = problem_description.find("Constraints")

        input_format = problem_description[input_start:output_start]
        output_format = problem_description[output_start:sample_input_start]
        # constraints = problem_description[constraints_start:].strip()

        # Strip HTML tags and extra whitespace
        input_format = strip_html_tags(input_format)
        output_format = strip_html_tags(output_format)

    return input_format, output_format, constraints


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


def generate_and_save_test_cases(problem_id, problem_description):
    """
    Use Ollama to generate input and output test cases based on the problem description,
    and save them as files under the problem_tests folder for the given problem_id.
    """
    
    # Define the directory for test cases
    problem_test_dir = f"problem_tests_generated/{problem_id}"
    os.makedirs(problem_test_dir, exist_ok=True)
    
    # Check how many test cases already exist
    # existing_inputs = [f for f in os.listdir(problem_test_dir) if f.startswith('input_')]
    # existing_test_cases_count = len(existing_inputs)
    
    # # If we already have 5 test cases, do nothing
    # if existing_test_cases_count >= 5:
    #     print(f"Test case limit reached for problem {problem_id}. No new test cases generated.")
    #     return
    
    # # Otherwise, generate only the remaining test cases needed
    remaining_test_cases = 20 #5 - existing_test_cases_count
    logging.info(f"Generating {remaining_test_cases} new test cases for problem {problem_id}...")

    # Extract input/output format and constraints from the problem description
    problem_details = extract_problem_details(problem_description)
    #input_format, output_format, constraints = extract_format_from_description(problem_description)

    # logging.info(f"Input Format: {input_format}")
    # logging.info(f"Output Format: {output_format}")
    # logging.info(f"Constraints: {constraints}")


    # # Prompt Ollama to generate the remaining test cases
    # response_old = ollama.chat(model="llama3.1", messages=[
    #     {
    #         "role": "user", 
    #         "content": f"""

    #         You are a coding and testing expert. Generate {remaining_test_cases} diverse test cases in JSON format for the following programming problem.
            
    #         Problem Description:
    #         {problem_description}

    #         Input Format:
    #         {input_format}

    #         Output Format:
    #         {output_format}
            
    #         The test cases should cover:
    #         1. **Basic Cases**: Simple scenarios to test basic functionality.

    #         Ensure the JSON output is strictly in the following format:
    #         {{
    #             "test_cases": [
    #                 {{
    #                     "input": "<formatted input>",
    #                     "expected_output": "<formatted output>"
    #                 }},
    #                 ...
    #             ]
    #         }}

    #         Guidelines:
    #         - Inputs and outputs should be valid and free of syntax errors.
    #         - Clearly separate each input-output pair.
    #         - Do not include any explanations or additional text.

    #         Validate the JSON before providing the response. Ensure no missing or incomplete data.
    #         """
    #     }
    # ])

    # # Prompt Ollama to generate the remaining test cases
    # response = ollama.chat(model="llama3.1", messages=[
    #     {
    #         "role": "system",
    #         "content": "You are a JSON generation assistant specializing in creating programming test cases."
    #     },
    #     {
    #         "role": "user", 
    #         "content": f"""

    #         Generate {remaining_test_cases} test cases for the following problem logically.
            
    #         Problem Description:
    #         {problem_description}

    #         Input Format:
    #         {input_format}

    #         Output Format:
    #         {output_format}

    #         Step 1: Understand the input/output structure.
    #         Step 2: Identify key cases
    #         Step 3: Generate inputs.
    #         Step 4: Compute outputs logically.
    #         Step 5: Compile the results into JSON format.

    #         Validation Checklist:
    #         1. Is the input format consistent with the problem description?
    #         2. Does the output match the logical computation of the input?
    #         3. Is the JSON output well-formed and free of syntax errors?    

    #         Ensure the JSON output is strictly in the following format:
    #         {{
    #             "test_cases": [
    #                 {{
    #                     "input": "<formatted input>",
    #                     "expected_output": "<formatted output>"
    #                 }},
    #                 ...
    #             ]
    #         }}

    #         Guidelines:
    #         - Inputs and outputs should be valid and free of syntax errors.
    #         - Clearly separate each input-output pair.
    #         - Do not include any explanations or additional text.

    #         Validate and ensure the test cases are logically consistent.
    #         """
    #     }
    # ], temperature=0.2)

    # GROQ implementation
    test_cases_prompt_file = f"{os.getcwd()}/scripts/prompts/test_cases_prompt.txt"
    with open(test_cases_prompt_file, 'r') as file:
        prompt = file.read()

    filled_prompt = prompt.format(
        problem_name=problem_details["problem_name"],
        description=problem_details["description"],
        model_input=problem_details["model_input"],
        model_output=problem_details["model_output"],
        sample_inputs=problem_details["sample_inputs"],
        sample_outputs=problem_details["sample_outputs"],
        num_cases=remaining_test_cases
    )

    client = Groq(
        api_key="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",
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": filled_prompt,
            }
        ],
        model="llama3-70b-8192",
        temperature=0.2,
    )

    # logging.info(response.choices[0].message.content)
    
    # Extract input/output from response
    #test_cases = response['message']['content']

    # test_cases = react_prompting(problem_id, problem_description, input_format, output_format)
    test_cases = response.choices[0].message.content

    # print(test_cases)
    inputs, outputs = extract_input_output_json(test_cases)  # Implement this to parse the test cases
    # return

    # Save the new test cases
    for i, (input_data, output_data) in enumerate(zip(inputs, outputs), start=1):
        input_file = os.path.join(problem_test_dir, f"input_{i}.txt")
        output_file = os.path.join(problem_test_dir, f"output_{i}.txt")
        
        # Save input
        with open(input_file, 'w') as f:
            f.write(input_data)
        
        # Save output
        with open(output_file, 'w') as f:
            f.write(output_data)
    
    logging.info(f"{remaining_test_cases} new test cases saved for problem {problem_id} in {problem_test_dir}")
    # return inputs, outputs

# def react_prompting(problem_id, problem_description, input_format, output_format, num_cases=20):
#     # steps = [
#     #     f"Analyze the problem and summarize the input/output structure and constraints. \n\nProblem Description:\n{problem_description} \nInput Format:\n{input_format}\nOutput Format:\n{output_format}\n",
#     #     # "Identify key test case categories (basic).", # Update this step - edge cases, negative, and complex cases).",
#     #     "Generate 20 basic inputs for the given problem.",
#     #     "Compute the corresponding outputs logically for all inputs. Do not include any explanations or additional text.",
#     #     "Validate all the test cases for logical consistency.",
#     #     f"""Compile the results into following JSON format. Do not include any explanations or additional text.
#     #         {{
#     #             "test_cases": [
#     #                 {{
#     #                     "input": "<formatted input>",
#     #                     "expected_output": "<formatted output>"
#     #                 }},
#     #                 ...
#     #             ]
#     #         }}"""
#     # ]

#     """
#     Generate test cases for a given problem using LangChain and Ollama.

#     Args:
#         problem_description (str): Problem description.
#         input_format (str): Input format description.
#         output_format (str): Output format description.
#         num_cases (int): Number of test cases to generate.

#     Returns:
#         dict: JSON object with generated test cases.
#     """
#     # Define steps for React prompting
#     steps = [
#         "Analyze the problem and summarize its constraints.",
#         f"Generate {num_cases} diverse test cases covering basic scenarios.",
#         "Validate and compile all test cases into a single JSON object."
#     ]

#     # Initialize conversation history and consolidated test cases
#     conversation_history = ""
#     final_test_cases = {"test_cases": []}

#     for step in steps:
#         # Generate response for the current step
#         response = conversation_chain.invoke(
#             # input={
#             #     "problem_description": problem_description,
#             #     "step": step
#             # }
#             problem_description=problem_description,
#             step=step,
#         )

#         logging.info(f"Step: {step}\nResponse: {response}")

#         # If final step, consolidate test cases
#         if step == "Validate and compile all test cases into a single JSON object.":
#             try:
#                 test_cases = json.loads(response)["test_cases"]
#                 final_test_cases["test_cases"].extend(test_cases)
#             except (json.JSONDecodeError, KeyError) as e:
#                 logging.error(f"Error parsing JSON: {e}")
#         else:
#             # Update conversation history to preserve context
#             conversation_history += f"Step: {step}\nResponse: {response}\n"

#     return final_test_cases


#     responses = []
#     test_cases = []
#     for stepIndex, step in enumerate(steps):
#         response = ollama.chat(
#             model="llama3.1",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a programming expert specializing in creating and validating test cases."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Step: {step}"
#                 }
#             ]
#         )
#         # responses.append(response)
#         logging.info(f"Step {stepIndex}: {step}")
#         logging.info(f"Response: {response['message']['content']}")

#         # # Validate the response
#         # if not validate_step(response["message"]["content"], stepIndex):
#         #     log_validation(stepIndex, response["message"]["content"], False)
#         #     response = ollama.chat(
#         #         model="llama3.1",
#         #         messages=[
#         #             {
#         #                 "role": "system",
#         #                 "content": "You are a programming expert specializing in creating and validating test cases."
#         #             },
#         #             {
#         #                 "role": "user",
#         #                 "content": f"Error in step: {step}. Please refine the output."
#         #             }
#         #         ]
#         #     )
#         #     responses.append(response)
        

#         if stepIndex == 5:
#             try:
#                 content = response["message"]["content"]
#                 # Try to parse the final JSON structure
#                 json_response = json.loads(content)
#                 if "test_cases" in json_response:
#                     test_cases = json_response["test_cases"]
#                     break  # Exit once the final test cases are extracted
#             except json.JSONDecodeError as e:
#                 logging.error(f"Invalid JSON format in response: {e}")
#                 continue

#     return {"test_cases": test_cases}

def validate_step(content, stepIndex):
    """
    Validate the intermediate output of a step in the React prompting workflow.

    Args:
        content (str): The output content to validate.
        stepIndex (str): The current step being validated.

    Returns:
        bool: True if the content is valid, False otherwise.
    """
    if stepIndex == 0:
        return validate_analysis(content)
    elif stepIndex == 1: # Update this step - negative, and complex cases).":
        return validate_categories(content)
    elif stepIndex == 2:
        return validate_inputs(content)
    elif stepIndex == 3:
        return validate_outputs(content)
    elif stepIndex == 4:
        return validate_test_cases(content)
    elif stepIndex == 5:
        return validate_json_format(content)
    return False

def validate_analysis(content):
    required_keywords = ["input", "output"]
    return all(keyword in content.lower() for keyword in required_keywords)

def validate_categories(content):
    return True
    required_categories = ["basic cases", "edge cases"] #, "negative cases", "complex cases"]
    return all(category.lower() in content.lower() for category in required_categories)


def validate_inputs(content):
    # Example: Ensure inputs follow a numeric or space-separated pattern
    return True 
    bool(re.search(r'\d+(\s+\d+)*', content))

def validate_outputs(content):
    # Example: Check for numerical outputs
    return True 
    bool(re.search(r'\d+(\.\d+)?(\s+\d+(\.\d+)?)*', content))

def validate_test_cases(content):
    try:
        test_cases = json.loads(content)
        for test_case in test_cases["test_cases"]:
            if "input" not in test_case or "expected_output" not in test_case:
                return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False

def validate_json_format(content):
    try:
        json_object = json.loads(content)
        return isinstance(json_object, dict) and "test_cases" in json_object
    except json.JSONDecodeError:
        return False
    
def log_validation(step, content, is_valid):
    if is_valid:
        logging.info(f"Step '{step}' passed validation.")
    else:
        logging.error(f"Step '{step}' failed validation: {content}")


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

def extract_input_output_json_2(test_cases_json):
    """
    Parse the JSON-formatted test cases from the Ollama response to extract inputs and outputs.
    JSON format:
    {
        "test_cases": [
            {
                "input": <input value>,
                "expected_output": <output value>
            },
            ...
        ]
    }
    """
    inputs = []
    outputs = []

    # Remove extra text before the JSON starts
    json_start = test_cases_json.find('{')
    if json_start != -1:
        clean_json = test_cases_json[json_start:]
        
        # Ensure the JSON ends correctly
        json_end = clean_json.rfind('}')
        if json_end != -1:
            clean_json = clean_json[:json_end + 1]

            # Parse the JSON response
            try:
                test_cases = json.loads(clean_json)
                
                # Extract inputs and outputs
                for test_case in test_cases.get("test_cases", []):
                    input_value = test_case.get("input")
                    expected_output = test_case.get("expected_output")

                    # print(f"Input: {input_value}")

                    # Ensure input_value is a string, regardless of its original format
                    if isinstance(input_value, list):
                        # Remove surrounding quotes if present
                        # input_value = [str(item).strip("'\"") for item in input_value if item]
                        input_value = [str(item) for item in input_value if str(item).strip()]
                        # print(f"Modified Input: {input_value}")
                        input_value = ' '.join(input_value)                    
                    else:
                        input_value = str(input_value).strip("'\"")

                    # print(f"Modified Input 2: {input_value}")

                    # Ensure expected_output is a string, regardless of its original format
                    if isinstance(expected_output, list):
                        expected_output = [str(item) for item in expected_output if str(item).strip()]
                        expected_output = ' '.join(expected_output)
                    else:
                        expected_output = str(expected_output).strip("'\"")

                    inputs.append(input_value)
                    outputs.append(expected_output)

            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON: {e}")
        else:
            logging.error("Error: JSON format is incomplete.")
    else:
        logging.error("Error: No valid JSON found in response.")

    return inputs, outputs
