# -*- coding: utf-8 -*-
# Python

"""Copyright (c) Alexander Fedotov.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from illuminations import utils


def get_weather(location):
    # print(f"Executing weather tool for location: {location}")
    return {"temperature": "72F", "condition": "Sunny"}


def chat_completion(messages=None, instructions=None, tools=None, **kwargs):
    """A continuation of text with a given context and instruction.
        kwargs:
            temperature     = 0 to 1.0
            top_p           = 0.0 to 1.0
            top_k           = The maximum number of tokens to consider when sampling.
            n               = 1 to ...
            max_tokens      = number of tokens
            stop            = ['stop']  array of up to 4 sequences
    """
    instruction         = kwargs.get('system_instruction', instructions)
    first_message       = [dict(role='system', content=instruction)] if instruction else []

    # contents can come in kwards or as an argument
    messages            = kwargs.get('messages', messages)

    # add contents and user text to the first (instruction) message
    first_message.extend(messages)
    instruction_and_contents = first_message

    payload = {
        'model':                    kwargs.get('model', utils.default_model),
        'messages':                 instruction_and_contents,
        # 'response_format':          kwargs.get('response_format',{'type': 'text'}),
        'temperature':              kwargs.get('temperature', 1),  # 0.0 to 2.0
        'max_tokens':               kwargs.get('max_tokens', 4096),
        'prompt_truncate_len':      kwargs.get('prompt_truncate_len', 100000),
        'n':                        kwargs.get('n', 1),
        'top_p':                    kwargs.get('top_p', 0.9),
        'top_k':                    kwargs.get('top_k', 10),
        'reasoning_effort':         kwargs.get('reasoning_effort', 'low'),  # 'low', 'medium', 'high'
        'reasoning_history':        kwargs.get('reasoning_history', None),  # 'disabled', 'interleaved', 'preserved'
        'stream':                   False
    }
    if tools:
        payload['tools'] = tools
        payload['parallel_tool_calls'] = True
        payload['tool_choice'] = 'auto'

    while True:
        result = utils.query(payload, '/chat/completions')
        completion_message = result['choices'][0]['message']
        instruction_and_contents.append(completion_message)
        thoughts = completion_message.get('reasoning_content', '')
        text = completion_message.get('content', '')
        function_calls = completion_message.get('tool_calls', [])

        if function_calls:
            # Call all requested functions and create response messages.
            for function_call in function_calls:
                call_id = function_call.get('id')
                func_def = function_call.get('function')
                func_name = func_def.get('name', '')

                # Look up tool by name in globals and caller frames
                func = utils.get_function(func_name)
                func_args = utils.get_func_args(func_def)
                result = utils.call_function(func, func_args)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result
                }
                instruction_and_contents.append(tool_message)
        else:
            break

    return thoughts, text


if __name__ == '__main__':
    ...
