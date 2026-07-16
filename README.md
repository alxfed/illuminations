### illuminations
A package for working with fireworks.ai without dependencies.
<pre>
  pip install illuminations
</pre>
Then:
```Python
  # Python
  import illuminations
```
After that:

```Python
from yaml import safe_load as yl
from illuminations.messages import message


kwargs = """  # this is a string in YAML format
  max_tokens:   32000
  stop_sequences:
    - STOP
    - "\nTitle"
  temperature:  1.0
  top_k:        10
  top_p:        0.5
  thinking:     
    type: enabled
    budget_tokens: 24576
    display: summarized
  tool_choice:
    type: auto
    disable_parallel_tool_use: false
"""

instruction = 'You are a helpful assistant. Important: Do not use markdown or lists in your responses.'

get_weather_tool_str = """ # YAML definition of a function
    name: get_weather
    description: Determine weather in a location
    input_schema:
      type: object
      properties:
        location:
          type: string
          description: The city and state, e.g. San Francisco, CA
      additionalProperties: true
      required:
        - location
    """

tools = [yl(get_weather_tool_str)]

msg = [{'role': 'user', 'content': 'What is the weather in Chicago, IL and Paris, France?'}]

thoughts, text = message(
    messages=msg,
    instructions=instruction,
    tools=tools,
    **yl(kwargs)
)
```
or
```Python
from yaml import safe_load as yl
from illuminations.chat import chat_complete as cc

kwargs = """  # this is a string in YAML format
  max_tokens:   32000
  stop_sequences:
    - STOP
    - "\nTitle"
  temperature:  1.0
  top_k:        10
  top_p:        0.5
  reasoning_effort: high
"""

instruction = 'You are a helpful assistant. Do not use markdown or lists in your responses.'

weather_tool = """ # YAML definition of a function (old format)
  type: function
  function:
    name: get_weather
    description: Determine weather in a location
    parameters:
      type: object
      properties:
        location:
          type: string
          description: The city and state, e.g. San Francisco, CA
      additionalProperties: false
      required:
        - location
"""

tools = [yl(weather_tool)]

msgs = [{'role': 'user', 'content': 'What is the weather in Chicago, IL and Paris, France?'}]

thoughts, text = cc(
    messages=msgs,
    instructions=instruction,
    tools=tools,
    **yl(kwargs)
)
```
or
```Python
```