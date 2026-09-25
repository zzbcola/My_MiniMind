单轮工具调用例子
messages = [
  {"role": "system",    "content": "# Tools ..."},
  {"role": "user",      "content": "帮我算 256 * 37"},
  {"role": "assistant", "content": '<tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>'},  ← 就是 new_text
  {"role": "tool",      "content": '{"result": "9472"}'}      ← 工具执行结果
]

多轮工具调用例子
[
  {"role": "system",    "content": "# Tools ..."},
  {"role": "user",      "content": "北京现在多热？56 是这个温度乘几？"},
  {"role": "assistant", "content": '<tool_call>{"name":"get_current_weather",...}</tool_call>'},
  {"role": "tool",      "content": '{"temperature":"28°C","condition":"晴"}'},
  {"role": "assistant", "content": '<tool_call>{"name":"calculate_math",...}</tool_call>'},
  {"role": "tool",      "content": '{"result":"56"}'},
]