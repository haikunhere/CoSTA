import json


def has_inline_tools(system):
    return "<tools>\n" in system and "\n</tools>" in system


def render_chat(tokenizer, messages, tools):
    messages = [dict(message) for message in messages]
    inline_tools = has_inline_tools(messages[0]["content"])
    if tools and not inline_tools:
        schema = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
        messages[0]["content"] += (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + schema + "\n</tools>\n\nFor each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
    for message in messages:
        if message["role"] == "tool":
            message.update(role="user", content="<tool_response>\n" + message["content"] + "\n</tool_response>")
    template_options = {} if inline_tools else {"enable_thinking": False}
    return tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, **template_options)


class Conversation:
    def __init__(self, tokenizer, messages, tools):
        self.tokenizer, self.tools = tokenizer, tools
        self.messages = list(messages)
        prompt = render_chat(tokenizer, self.messages, tools)
        self.ids = tokenizer.encode(prompt, add_special_tokens=False)
        self.generated = [0] * len(self.ids)

    def append_sample(self, ids):
        raw_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.ids.extend(ids)
        self.generated.extend([1] * len(ids))
        text = raw_text
        if ids[-1] == self.tokenizer.eos_token_id:
            text = text.removesuffix(self.tokenizer.eos_token)
        self.messages.append(dict(role="assistant", content=text))
        return raw_text

    def observe(self, text, *, role="tool"):
        self.messages.append(dict(role=role, content=text))
        marked = list(self.messages)
        marked[-2] = {**marked[-2], "content": marked[-2]["content"] + "\0"}
        _, added = render_chat(self.tokenizer, marked, self.tools).rsplit("\0", 1)
        if self.ids[-1] == self.tokenizer.eos_token_id:
            added = added.removeprefix(self.tokenizer.eos_token)
        ids = self.tokenizer.encode(added, add_special_tokens=False)
        self.ids.extend(ids)
        self.generated.extend([0] * len(ids))
