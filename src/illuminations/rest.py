# -*- coding: utf-8 -*-
# Python

"""Copyright (c) Alexander Fedotov.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from os import environ
import json
import urllib.request
import urllib.error
import requests
from .adapter import decode


api_key                 = environ.get('FIREWORKS_API_KEY')
api_base                = environ.get('FIREWORKS_BASE_URL', 'https://api.fireworks.ai/inference/v1')
default_model = environ.get('FIREWORKS_MODEL','accounts/fireworks/models/gpt-oss-120b')

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + api_key,
    "User-Agent": "illuminations"
}


def get_function(func_name):
    # Look up tool by name in globals
    func = globals().get(func_name)
    # Look up in the caller frames
    if not func:
        import inspect
        frame = inspect.currentframe().f_back
        while frame:
            if func_name in frame.f_globals:
                func = frame.f_globals[func_name]
                break
            frame = frame.f_back
    return func


def query(payload):
    # Convert data dictionary to JSON and encode it to bytes
    data_bytes = json.dumps(payload).encode('utf-8')
    # Create the Request object
    req = urllib.request.Request(
        f'{api_base}/chat/completions',
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


def continuation(text=None, contents=None, instruction=None, recorder=None, **kwargs):
    """A continuation of text with a given context and instruction.
        kwargs:
            temperature     = 0 to 1.0
            top_p           = 0.0 to 1.0
            top_k           = The maximum number of tokens to consider when sampling.
            n               = 1 to ...
            max_tokens      = number of tokens
            stop            = ['stop']  array of up to 4 sequences
    """
    instruction         = kwargs.get('system_instruction', instruction)
    first_message       = [dict(role='system', content=instruction)] if instruction else []

    # contents can come in kwards or as an argument
    contents            = kwargs.get('contents', contents)

    # if there is a recorded log of the previous conversation
    if recorder and not contents:
        contents = recorder.log.copy()

    # now add the incoming human text
    human_says = dict(role='user', content=text)
    if text and not contents:
        contents = [human_says]
    else:
        contents.append(human_says)

    # add contents and user text to the first (instruction) message
    first_message.extend(contents)
    instruction_and_contents = first_message

    json_data = {
        'model':                    kwargs.get('model', default_model),
        'messages':                 instruction_and_contents,
        'response_format':          kwargs.get('response_format',{'type': 'text'}),
        'temperature':              kwargs.get('temperature', 1),  # 0.0 to 2.0
        'max_tokens':               kwargs.get('max_tokens', 4096),
        'prompt_truncate_len':      kwargs.get('prompt_truncate_len', 100000),
        'n':                        kwargs.get('n', 1),
        'top_p':                    kwargs.get('top_p', 0.9),
        'top_k':                    kwargs.get('top_k', 10),
        'reasoning_effort':         kwargs.get('reasoning_effort', 'low'),  # 'low', 'medium', 'high'
        'reasoning_history':        kwargs.get('reasoning_history', None),  # 'disabled', 'interleaved', 'preserved'
        'tools':                    kwargs.get('tools', {}),
        'parallel_tool_calls':      kwargs.get('parallel_tool_calls', True),
        'stream':                   False
    }

    try:
        response = requests.post(
            url=f'{api_base}/chat/completions',
            headers=headers,
            json=json_data,
        )
        if response.status_code == requests.codes.ok:
            output = response.json()
            answer = decode(human_says, output, recorder)
        else:
            print(f'Request status code: {response.status_code}')
            return None

    except Exception as e:
        print(f'Unable to generate continuation of the text, {e}')
        return None

    return answer


def completion(text, **kwargs):
    """A completions endpoint call through requests.
        kwargs:
            temperature     = 0 to 1.0
            top_p           = 0.0 to 1.0
            n               = 1 to ...
            best_of         = 4
            frequency_penalty = -2.0 to 2.0
            presence_penalty = -2.0 to 2.0
            max_tokens      = number of tokens
            logprobs        = number up to 5
            stop            = ["stop"]  array of up to 4 sequences
            logit_bias      = map token: bias -1.0 to 1.0 (restrictive -100 to 100)

    Use this method as follows:
    ..  code-block:: python
        res = aFunction(something, goes, in)
        print(res.avalue)
    """
    responses = []
    json_data = {
        "model":            kwargs.get("model", default_model),
        "prompt":           kwargs.get("prompt", text),
        "suffix":           kwargs.get("suffix", None),
        "max_tokens":       kwargs.get("max_tokens", 5),
        "n":                kwargs.get("n", 1),
        "best_of":          kwargs.get("best_of", 1),
        "stop":             kwargs.get("stop_sequences", ["stop"]),
        "seed":             kwargs.get("seed", None),
        "frequency_penalty":kwargs.get("frequency_penalty", None),
        "presence_penalty": kwargs.get("presence_penalty", None),
        "logit_bias":       kwargs.get("logit_bias", None),
        "logprobs":         kwargs.get("logprobs", None),
        "top_logprobs":     kwargs.get("top_logprobs", None),
        "temperature":      kwargs.get("temperature", 0.5),
        "top_p":            kwargs.get("top_p", 0.5),
        "user":             kwargs.get("user", None)
    }

    try:
        response = requests.post(
            f"{api_base}/completions",
            headers=headers,
            json=json_data,
        )
        if response.status_code == requests.codes.ok:
            for choice in response.json()['choices']:
                responses.append(choice)
        else:
            print(f"Request status code: {response.status_code}")
        return responses
    except Exception as e:
        print("Unable to generate Completions response")
        print(f"Exception: {e}")
        return responses


def respond(messages=None, instructions=None, tools=None, **kwargs):
    """ All parameters should be in kwargs, but they are optional
    """
    # Receive the instruction
    instruction = kwargs.get('system_instruction', instructions)
    first_message = [dict(role='system', content=instruction)] if instruction else []

    # add contents and user text to the first (instruction) message
    first_message.extend(messages)
    instruction_and_contents = first_message

    # Define the initial payload
    payload = {
        "model":            kwargs.get("model", default_model),
        "messages":         instruction_and_contents,
        "max_tokens":       kwargs.get("max_tokens", 132000),
        "reasoning_effort": "max",
    }
    # Tools if there are some
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'

    complete_thoughts = ''

    while True:
        # Query the API
        result = query(payload)
        # id of the cached response can be here some day
        # response_id = result['id']
        completion_message = result['choices'][0]['message']
        instruction_and_contents.append(completion_message)
        thoughts = completion_message.get('reasoning_content', '')
        complete_thoughts += '\n\n' + thoughts + '\n\n'
        text = completion_message.get('content', '')
        function_calls = completion_message.get('tool_calls', [])

        if function_calls:
            # Call all requested functions and create response messages.
            for function_call in function_calls:
                call_id = function_call.get('id')
                func = function_call.get('function') # Old format of function.
                func_name = func.get('name')
                func_args_str = func.get('arguments', '{}')
                try:
                    if isinstance(func_args_str, str):
                        func_args = json.loads(func_args_str)
                    else:
                        func_args = func_args_str
                except Exception as e:
                    func_args = {}
                    print(f"Error parsing tool arguments for {func_name}: {e}")

                func = get_function(func_name)

                if func and callable(func):  # not a duplicate
                    try:
                        function_result = func(**func_args)
                        if isinstance(function_result, (dict, list)):
                            result = json.dumps(function_result)
                        else:
                            result = str(function_result)
                    except Exception as e:
                        result = f"Error executing tool {func_name}: {str(e)}"
                        print(result)
                else:
                    result = f"Error: Tool function {func_name} not found."
                    print(result)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result
                }
                # Add the response
                instruction_and_contents.append(tool_message)
        else:
            break
    return thoughts, text


if __name__ == '__main__':
    ...
