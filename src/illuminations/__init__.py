# -*- coding: utf-8 -*-
# Python

"""Copyright (c) Alexander Fedotov.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from .utils import (query,
                    headers,
                    get_function,
                    get_func_args,
                    call_function,
                    decode_output,
                    default_model)
from .chat import chat_complete
from .completion import complete
from .responses import respond

__all__ = [
    "chat_complete",
    "complete",
    "respond",
    # The API calling
    "query",
    "headers",
    # and all the function calling stuff
    "get_function",
    "get_func_args",
    "call_function",
    "decode_output",
    "default_model"
]