# -*- coding: utf-8 -*-
# Python

"""Copyright (c) Alexander Fedotov.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import urllib.request
import urllib.error
import json
from os import environ

api_key = environ.get("FIREWORKS_API_KEY", '')
default_model = environ.get("FIREWORKS_DEFAULT_MODEL", 'accounts/fireworks/models/gpt-oss-120b')
api_base = environ.get("FIREWORKS_API_BASE", 'https://api.fireworks.ai/inference/v1')


# Set the mandatory headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "User-Agent": "illuminations"
}


def machine_shop(machine_name, task, context, **additional):
    print('Received: ', machine_name, task, context)
    response = f"{machine_name}: I don't know. I can't help you with this."
    return response


def get_weather(location):
    # print(f"Executing weather tool for location: {location}")
    return {"temperature": "72F", "condition": "Sunny"}


def query(payload):
    # Convert data dictionary to JSON and encode it to bytes
    data_bytes = json.dumps(payload).encode('utf-8')
    # Create the Request object
    req = urllib.request.Request(
        f'{api_base}/responses',
        data=data_bytes,
        headers=headers,
        method="POST")
    # Try to query
    try:
        # Execute the request
        with urllib.request.urlopen(req, timeout=3000) as response:
            response_data = response.read().decode('utf-8')
            output = json.loads(response_data)
        return output

    except urllib.error.HTTPError as e:
        # Handle HTTP errors (e.g., 401 Unauthorized, 400 Bad Request)
        error_info = e.read().decode('utf-8', errors='ignore')
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Error Details: {error_info}")
        return {}

    except urllib.error.URLError as e:
        # Handle network/connection errors
        print(f"Failed to reach the server: {e.reason}")
        return {}


def decode_output(output):
    # Parse the result
    text = ''; thoughts = ''
    for part in output:
        part_type = part.get('type', None)
        if part_type == 'message':
            text = " ".join([chunk['text'] for chunk in part['content'] if chunk['type'] == 'output_text'])
        elif part_type == 'reasoning':
            thoughts = " ".join([chunk['text'] for chunk in part['summary'] if chunk['type'] == 'summary_text'])
    function_calls = [part for part in output if part['type'] == 'function_call']
    return thoughts, text, function_calls


def respond(messages=None, instructions=None, tools=None, **kwargs):
    """ All parameters should be in kwargs, but they are optional
    """
    # Receive the instruction
    instruction = kwargs.get('system_instruction', instructions)

    # Define the initial payload
    payload = {
        "model":            kwargs.get("model", default_model),
        "instructions":     instruction,
        "input":            messages,
        "previous_response_id": kwargs.get("previous_response_id", None),
        "max_output_tokens": kwargs.get("max_tokens", 132000),
        "prompt_cache_retention": "in_memory",
        "include": ["reasoning.encrypted_content"],
        "reasoning": {
            "effort": "high",
            "summary": "detailed"
        }
    }
    # Tools if there are some
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'

    while True:
        # Query the API
        result = query(payload)
        # id of the response
        response_id = result['id']
        thoughts, text, function_calls = decode_output(result.get('output', {}))

        if function_calls:
            function_outputs_messages = []
            for function_call in function_calls:
                call_id = function_call.get('call_id')
                func_name = function_call.get('name')
                func_args_str = function_call.get('arguments', '{}')
                try:
                    if isinstance(func_args_str, str):
                        func_args = json.loads(func_args_str)
                    else:
                        func_args = func_args_str
                except Exception as e:
                    func_args = {}
                    print(f"Error parsing tool arguments for {func_name}: {e}")

                # Look up tool by name in globals and caller frames
                func = globals().get(func_name)
                if not func:
                    import inspect
                    frame = inspect.currentframe().f_back
                    while frame:
                        if func_name in frame.f_globals:
                            func = frame.f_globals[func_name]
                            break
                        frame = frame.f_back

                if func and callable(func):
                    try:
                        tool_result = func(**func_args)
                        if isinstance(tool_result, (dict, list)):
                            result = json.dumps(tool_result)
                        else:
                            result = str(tool_result)
                    except Exception as e:
                        result = f"Error executing tool {func_name}: {str(e)}"
                        print(result)
                else:
                    result = f"Error: Tool function {func_name} not found."
                    print(result)

                tool_message = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result
                }
                function_outputs_messages.append(tool_message)

            # Now that all responses have been gathered
            # we can change the payload and send them back
            payload['previous_response_id'] = response_id
            payload['input'] = function_outputs_messages
        else:
            break

    return thoughts, text


if __name__ == "__main__":
    ...