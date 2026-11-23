"""A ReAct-style tool loop: model -> tools -> model ... -> END, with a human approval stop.

Runs offline: ``ScriptedModel`` stands in for an LLM so the example is repeatable. Swap
``call_model`` for a real chat API call (Claude, Bedrock, ...) that returns the same shape.

    python3 examples/react_agent.py
"""
import ast
import operator
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_graph import END, MemoryCheckpointer, StateGraph, add, append  # noqa: E402

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def calculator(expression: str) -> float:
    """Safe arithmetic (no eval)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {expression}")
    return ev(ast.parse(expression, mode="eval").body)


TOOLS = {"calculator": calculator}


class ScriptedModel:
    """Returns pre-written turns in order, like a recorded LLM conversation."""

    def __init__(self, turns):
        self.turns = list(turns)

    def __call__(self, messages):
        return self.turns.pop(0)


def build(model, checkpointer=None, approve_tools=False):
    def call_model(state):
        reply = model(state["messages"])
        return {"messages": [reply], "llm_calls": 1}

    def run_tools(state):
        results = []
        for call in state["messages"][-1]["tool_calls"]:
            try:
                out = TOOLS[call["name"]](**call["args"])
                results.append({"role": "tool", "name": call["name"], "content": str(out)})
            except Exception as exc:  # report errors back to the model, do not crash
                results.append({"role": "tool", "name": call["name"],
                                "content": f"error: {exc}", "is_error": True})
        return {"messages": results}

    def route(state):
        return "tools" if state["messages"][-1].get("tool_calls") else "done"

    g = StateGraph(reducers={"messages": append, "llm_calls": add})
    g.add_node("model", call_model, retries=2)
    g.add_node("tools", run_tools)
    g.set_entry("model")
    g.add_conditional_edges("model", route, {"tools": "tools", "done": END})
    g.add_edge("tools", "model")
    return g.compile(max_steps=10, checkpointer=checkpointer,
                     interrupt_before=["tools"] if approve_tools else [])


DEMO_TURNS = [
    {"role": "assistant", "content": "",
     "tool_calls": [{"name": "calculator", "args": {"expression": "1200 * 0.15"}}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"name": "calculator", "args": {"expression": "1200 - 180"}}]},
    {"role": "assistant", "content": "15% of 1200 is 180, so the discounted price is 1020."},
]

if __name__ == "__main__":
    question = {"messages": [{"role": "user", "content": "What is 1200 after a 15% discount?"}]}
    app = build(ScriptedModel(DEMO_TURNS))
    for step in app.stream(question):
        print(f"step {step.index}: {step.node:<6} -> {step.next_node}")
    print("answer:", step.state["messages"][-1]["content"])
    print("llm calls:", step.state["llm_calls"])

    print("\n# human-in-the-loop: stop before every tool call")
    cp = MemoryCheckpointer()
    app = build(ScriptedModel(DEMO_TURNS), checkpointer=cp, approve_tools=True)
    res = app.run(question, thread_id="demo")
    while res.status == "interrupted":
        call = res.state["messages"][-1]["tool_calls"][0]
        print(f"paused before '{res.next_node}': approve {call['name']}({call['args']})? yes")
        res = app.resume("demo")
    print("answer:", res.state["messages"][-1]["content"])
    print()
    print(build(ScriptedModel([])).graph.to_mermaid())
