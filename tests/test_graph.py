import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from agent_graph import (END, START, GraphError, GraphRecursionError, JsonFileCheckpointer,  # noqa: E402
                         MemoryCheckpointer, NodeError, StateGraph, add, append, merge_dict)
import react_agent  # noqa: E402


def linear():
    g = StateGraph(reducers={"log": append})
    g.add_node("a", lambda s: {"log": "a", "x": 1})
    g.add_node("b", lambda s: {"log": "b", "x": s["x"] + 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


class TestReducers(unittest.TestCase):
    def test_append(self):
        self.assertEqual(append(None, 1), [1])
        self.assertEqual(append([1], [2, 3]), [1, 2, 3])
        self.assertEqual(append((1,), 2), [1, 2])

    def test_add_and_merge(self):
        self.assertEqual(add(None, 3), 3)
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(merge_dict({"a": 1}, {"b": 2}), {"a": 1, "b": 2})


class TestRun(unittest.TestCase):
    def test_linear_run(self):
        res = linear().compile().run({})
        self.assertEqual(res.status, "done")
        self.assertEqual(res.path, ["a", "b"])
        self.assertEqual(res.state, {"log": ["a", "b"], "x": 2})

    def test_input_state_not_mutated(self):
        start = {"log": ["start"]}
        linear().compile().invoke(start)
        self.assertEqual(start, {"log": ["start"]})

    def test_node_cannot_mutate_state_in_place(self):
        def sneaky(s):
            s["log"].append("hacked")
            return None
        g = StateGraph()
        g.add_node("n", sneaky).set_entry("n").add_edge("n", END)
        self.assertEqual(g.compile().invoke({"log": []}), {"log": []})

    def test_node_must_return_dict(self):
        g = StateGraph()
        g.add_node("n", lambda s: "oops").set_entry("n").add_edge("n", END)
        with self.assertRaises(TypeError):
            g.compile().invoke({})

    def test_conditional_loop(self):
        g = StateGraph(reducers={"n": add})
        g.add_node("inc", lambda s: {"n": 1})
        g.set_entry("inc")
        g.add_conditional_edges("inc", lambda s: "again" if s["n"] < 3 else "stop",
                                {"again": "inc", "stop": END})
        res = g.compile().run({"n": 0})
        self.assertEqual(res.state["n"], 3)
        self.assertEqual(res.path, ["inc"] * 3)

    def test_router_without_mapping(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).add_node("b", lambda s: {"hit": True})
        g.set_entry("a").add_conditional_edges("a", lambda s: "b").add_edge("b", END)
        self.assertTrue(g.compile().invoke({})["hit"])

    def test_router_bad_key(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry("a")
        g.add_conditional_edges("a", lambda s: "nope", {"done": END})
        with self.assertRaises(GraphError):
            g.compile().invoke({})

    def test_router_unknown_node(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry("a").add_conditional_edges("a", lambda s: "zzz")
        with self.assertRaises(GraphError):
            g.compile().invoke({})

    def test_max_steps(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry("a").add_edge("a", "a")
        with self.assertRaises(GraphRecursionError):
            g.compile(max_steps=5).invoke({})

    def test_stream_steps(self):
        steps = list(linear().compile().stream({}))
        self.assertEqual([(s.index, s.node, s.next_node) for s in steps],
                         [(1, "a", "b"), (2, "b", END)])
        self.assertEqual(steps[0].update, {"log": "a", "x": 1})
        self.assertEqual(steps[1].state["x"], 2)


class TestRetries(unittest.TestCase):
    def test_retry_then_success(self):
        calls = {"n": 0}

        def flaky(s):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("slow model")
            return {"ok": True}
        g = StateGraph()
        g.add_node("f", flaky, retries=2, retry_on=[TimeoutError]).set_entry("f").add_edge("f", END)
        res = g.compile().run({})
        self.assertTrue(res.state["ok"])
        self.assertEqual(res.steps[0].attempts, 3)

    def test_retry_exhausted(self):
        def bad(s):
            raise TimeoutError("down")
        g = StateGraph()
        g.add_node("f", bad, retries=1).set_entry("f").add_edge("f", END)
        with self.assertRaises(NodeError) as ctx:
            g.compile().invoke({})
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIsInstance(ctx.exception.cause, TimeoutError)

    def test_non_retryable_error_not_retried(self):
        calls = {"n": 0}

        def bad(s):
            calls["n"] += 1
            raise KeyError("bug")
        g = StateGraph()
        g.add_node("f", bad, retries=3, retry_on=[TimeoutError]).set_entry("f").add_edge("f", END)
        with self.assertRaises(KeyError):
            g.compile().invoke({})
        self.assertEqual(calls["n"], 1)


class TestValidation(unittest.TestCase):
    def test_no_entry(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).add_edge("a", END)
        with self.assertRaises(GraphError):
            g.compile()

    def test_dangling_node(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry("a")
        with self.assertRaisesRegex(GraphError, "no outgoing edge"):
            g.compile()

    def test_unknown_target(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry("a").add_edge("a", "b")
        with self.assertRaisesRegex(GraphError, "unknown node 'b'"):
            g.compile()

    def test_unreachable(self):
        g = linear()
        g.add_node("orphan", lambda s: None).add_edge("orphan", END)
        with self.assertRaisesRegex(GraphError, "unreachable.*orphan"):
            g.compile()

    def test_duplicates_and_reserved(self):
        g = StateGraph()
        g.add_node("a", lambda s: None)
        with self.assertRaises(GraphError):
            g.add_node("a", lambda s: None)
        with self.assertRaises(GraphError):
            g.add_node(END, lambda s: None)
        g.add_edge("a", END)
        with self.assertRaises(GraphError):
            g.add_conditional_edges("a", lambda s: END)

    def test_bad_interrupt_node(self):
        with self.assertRaises(GraphError):
            linear().compile(interrupt_before=["zzz"])

    def test_mermaid(self):
        text = linear().to_mermaid()
        self.assertIn("a --> b", text)
        self.assertIn("b --> __end__([end])", text)


class TestCheckpoints(unittest.TestCase):
    def test_interrupt_and_resume_with_update(self):
        cp = MemoryCheckpointer()
        app = linear().compile(checkpointer=cp, interrupt_before=["b"])
        res = app.run({}, thread_id="t1")
        self.assertEqual((res.status, res.next_node), ("interrupted", "b"))
        self.assertEqual(res.state["log"], ["a"])
        res = app.resume("t1", update={"x": 10, "log": "human-ok"})
        self.assertEqual(res.status, "done")
        self.assertEqual(res.state, {"log": ["a", "human-ok", "b"], "x": 11})
        self.assertEqual(cp.get("t1")["status"], "done")

    def test_resume_done_thread_is_noop(self):
        cp = MemoryCheckpointer()
        app = linear().compile(checkpointer=cp)
        first = app.run({}, thread_id="t")
        again = app.resume("t")
        self.assertEqual(again.state, first.state)
        self.assertEqual(again.steps, [])

    def test_resume_after_crash(self):
        state = {"fail": True}

        def b(s):
            if state["fail"]:
                raise RuntimeError("crash")
            return {"x": s["x"] + 1, "log": "b"}
        g = StateGraph(reducers={"log": append})
        g.add_node("a", lambda s: {"log": "a", "x": 1}).add_node("b", b)
        g.set_entry("a").add_edge("a", "b").add_edge("b", END)
        with tempfile.TemporaryDirectory() as d:
            app = g.compile(checkpointer=JsonFileCheckpointer(d))
            with self.assertRaises(NodeError):
                app.run({}, thread_id="job-1")
            saved = JsonFileCheckpointer(d).get("job-1")
            self.assertEqual((saved["next_node"], saved["step"]), ("b", 1))
            state["fail"] = False
            res = app.resume("job-1")
            self.assertEqual(res.path, ["b"])
            self.assertEqual(res.state, {"log": ["a", "b"], "x": 2})

    def test_resume_needs_checkpoint(self):
        app = linear().compile(checkpointer=MemoryCheckpointer())
        with self.assertRaises(GraphError):
            app.resume("missing")
        with self.assertRaises(GraphError):
            linear().compile().resume("x")

    def test_unsafe_thread_id(self):
        with tempfile.TemporaryDirectory() as d:
            cp = JsonFileCheckpointer(d)
            for bad in ["../x", "a/b", "", ".."]:
                with self.assertRaises(ValueError):
                    cp.put(bad, {})
            self.assertEqual(os.listdir(d), [])

    def test_memory_history(self):
        cp = MemoryCheckpointer()
        linear().compile(checkpointer=cp).run({}, thread_id="h")
        self.assertEqual([c["step"] for c in cp.history["h"]], [1, 2])


class TestReactExample(unittest.TestCase):
    def test_tool_loop(self):
        app = react_agent.build(react_agent.ScriptedModel(react_agent.DEMO_TURNS))
        res = app.run({"messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(res.path, ["model", "tools", "model", "tools", "model"])
        self.assertEqual(res.state["llm_calls"], 3)
        tool_msgs = [m["content"] for m in res.state["messages"] if m["role"] == "tool"]
        self.assertEqual(tool_msgs, ["180.0", "1020"])

    def test_tool_error_goes_back_to_model(self):
        turns = [{"role": "assistant", "tool_calls": [
                     {"name": "calculator", "args": {"expression": "__import__('os')"}}]},
                 {"role": "assistant", "content": "Sorry, I cannot compute that."}]
        res = react_agent.build(react_agent.ScriptedModel(turns)).run({"messages": []})
        tool = [m for m in res.state["messages"] if m["role"] == "tool"][0]
        self.assertTrue(tool["is_error"])
        self.assertEqual(res.status, "done")

    def test_human_approval(self):
        cp = MemoryCheckpointer()
        app = react_agent.build(react_agent.ScriptedModel(react_agent.DEMO_TURNS),
                                checkpointer=cp, approve_tools=True)
        res = app.run({"messages": []}, thread_id="hitl")
        stops = 0
        while res.status == "interrupted":
            stops += 1
            res = app.resume("hitl")
        self.assertEqual(stops, 2)
        self.assertIn("1020", res.state["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
