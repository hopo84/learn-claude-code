#!/usr/bin/env python
"""
v0_bash_agent.py - Mini Claude Code: Bash is All You Need (~50 lines core)

Core Philosophy: "Bash is All You Need"
======================================
This is the ULTIMATE simplification of a coding agent. After building v1-v3,
we ask: what is the ESSENCE of an agent?

The answer: ONE tool (bash) + ONE loop = FULL agent capability.

Why Bash is Enough:
------------------
Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep                  |
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python v0_bash_agent.py "task"         |

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.

How Subagents Work:
------------------
    Main Agent
      |-- bash: python v0_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting

Usage:
    # Interactive mode
    python v0_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python v0_bash_agent.py "explore src/ and summarize"
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess
import sys
import os
import json
import urllib.request
import urllib.error

load_dotenv(override=True)

# Model selection
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")

# OpenRouter support (OpenAI-compatible endpoint)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY and "openrouter.ai" in ANTHROPIC_BASE_URL:
    # Allow using ANTHROPIC_API_KEY as fallback when base_url points to OpenRouter
    OPENROUTER_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE")
USE_OPENROUTER = bool(OPENROUTER_API_KEY) or ("openrouter.ai" in ANTHROPIC_BASE_URL)
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID")

# Initialize Anthropic client (uses ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL env vars)
client = None if USE_OPENROUTER else Anthropic(base_url=ANTHROPIC_BASE_URL or None)

# The ONE tool that does everything
# Notice how the description teaches the model common patterns AND how to spawn subagents
TOOL = [{
    "name": "bash",
    "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v0_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }
}]

# System prompt teaches the model HOW to use bash effectively
# Notice the subagent guidance - this is how we get hierarchical task decomposition
SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python v0_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""


def _openrouter_request(payload, path="/chat/completions"):
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {err.code}: {body}") from None


_OPENROUTER_MODELS_CACHE = None


def _openrouter_list_models():
    global _OPENROUTER_MODELS_CACHE
    if _OPENROUTER_MODELS_CACHE is not None:
        return _OPENROUTER_MODELS_CACHE
    data = _openrouter_request(None, path="/models")
    models = [item.get("id", "") for item in data.get("data", [])]
    _OPENROUTER_MODELS_CACHE = [m for m in models if m]
    return _OPENROUTER_MODELS_CACHE


def _resolve_openrouter_model(requested):
    if not requested:
        requested = "anthropic/claude-sonnet-4.5"

    if not requested.startswith(("anthropic/", "openai/", "google/", "meta/")):
        candidate = f"anthropic/{requested}"
    else:
        candidate = requested

    try:
        models = _openrouter_list_models()
    except RuntimeError:
        return candidate

    if candidate in models:
        return candidate

    if requested in models:
        return requested

    # Prefer Claude Sonnet if possible, else any Claude, else first model.
    for m in models:
        if "anthropic/" in m and "claude" in m and "sonnet" in m:
            return m
    for m in models:
        if "anthropic/" in m and "claude" in m:
            return m
    return models[0] if models else candidate


def chat_openrouter(prompt, history=None):
    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    model_id = _resolve_openrouter_model(OPENROUTER_MODEL_ID or MODEL)

    openai_tools = [{
        "type": "function",
        "function": {
            "name": TOOL[0]["name"],
            "description": TOOL[0]["description"],
            "parameters": TOOL[0]["input_schema"]
        }
    }]

    while True:
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": SYSTEM}] + history,
            "tools": openai_tools,
            "tool_choice": "auto",
            "max_tokens": 8000
        }
        response = _openrouter_request(payload)
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        history.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls
        })

        if not tool_calls:
            return message.get("content", "")

        results = []
        for call in tool_calls:
            try:
                args = json.loads(call["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            cmd = args.get("command", "")
            print(f"\033[33m$ {cmd}\033[0m")
            try:
                out = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.getcwd()
                )
                output = out.stdout + out.stderr
            except subprocess.TimeoutExpired:
                output = "(timeout after 300s)"

            print(output or "(empty)")
            results.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output[:50000]
            })

        history.extend(results)


def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.

    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)

    Returns:
        Final text response from the model
    """
    if USE_OPENROUTER:
        return chat_openrouter(prompt, history)

    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    while True:
        # 1. Call the model with tools
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=history,
            tools=TOOL,
            max_tokens=8000
        )

        # 2. Build assistant message content (preserve both text and tool_use blocks)
        content = []
        for block in response.content:
            if hasattr(block, "text"):
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        history.append({"role": "assistant", "content": content})

        # 3. If model didn't call tools, we're done
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        # 4. Execute each tool call and collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                cmd = block.input["command"]
                print(f"\033[33m$ {cmd}\033[0m")  # Yellow color for commands

                try:
                    out = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=os.getcwd()
                    )
                    output = out.stdout + out.stderr
                except subprocess.TimeoutExpired:
                    output = "(timeout after 300s)"

                print(output or "(empty)")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output[:50000]  # Truncate very long outputs
                })

        # 5. Append results and continue the loop
        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        # This is how parent agents spawn children via bash
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        history = []
        while True:
            try:
                query = input("\033[36m>> \033[0m")  # Cyan prompt
            except (EOFError, KeyboardInterrupt):
                break
            if query in ("q", "exit", ""):
                break
            print(chat(query, history))
