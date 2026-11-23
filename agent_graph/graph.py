"""A small state-graph runner for LLM agents.

Nodes are plain functions ``fn(state) -> dict | None``. The returned dict is merged into the
state with per-key reducers (overwrite by default). Edges are fixed (``add_edge``) or decided
at run time by a router function (``add_conditional_edges``). The compiled graph can stream
every step, stop before chosen nodes for human review, save a checkpoint after every step and
resume a run later from that checkpoint.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type

START = "__start__"
END = "__end__"

State = Dict[str, Any]
Reducer = Callable[[Any, Any], Any]


class GraphError(Exception):
    """The graph definition is invalid."""


class GraphRecursionError(RuntimeError):
    """The run took more steps than ``max_steps`` (usually an agent loop that never ends)."""


class NodeError(RuntimeError):
    """A node failed after all its retries."""

    def __init__(self, node: str, attempts: int, cause: BaseException):
        super().__init__(f"node '{node}' failed after {attempts} attempt(s): {cause!r}")
        self.node = node
        self.attempts = attempts
        self.cause = cause


# ---------------------------------------------------------------- reducers
def overwrite(old: Any, new: Any) -> Any:
    return new


def append(old: Any, new: Any) -> list:
    """List reducer: ``[1] + 2 -> [1, 2]``, ``[1] + [2, 3] -> [1, 2, 3]``."""
    base = list(old) if old is not None else []
    if isinstance(new, (list, tuple)):
        return base + list(new)
    return base + [new]


def add(old: Any, new: Any) -> Any:
    """Numeric reducer, e.g. for token or cost counters."""
    return (old or 0) + new


def merge_dict(old: Any, new: Any) -> dict:
    out = dict(old or {})
    out.update(new or {})
    return out


# ---------------------------------------------------------------- data types
@dataclass
class _Node:
    name: str
    fn: Callable[[State], Optional[State]]
    retries: int = 0
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)
    backoff: float = 0.0


@dataclass
class _Branch:
    router: Callable[[State], str]
    mapping: Optional[Dict[str, str]] = None

    def targets(self) -> Optional[List[str]]:
        return list(self.mapping.values()) if self.mapping is not None else None


@dataclass
class Step:
    """One executed node, as yielded by ``CompiledGraph.stream``."""

    index: int
    node: str
    update: State
    next_node: str
    state: State
    attempts: int = 1
    seconds: float = 0.0


@dataclass
class RunResult:
    state: State
    status: str  # "done" or "interrupted"
    next_node: Optional[str]
    steps: List[Step] = field(default_factory=list)
    thread_id: Optional[str] = None

    @property
    def path(self) -> List[str]:
        return [s.node for s in self.steps]


# ---------------------------------------------------------------- builder
class StateGraph:
    def __init__(self, reducers: Optional[Dict[str, Reducer]] = None):
        self.reducers: Dict[str, Reducer] = dict(reducers or {})
        self.nodes: Dict[str, _Node] = {}
        self.edges: Dict[str, str] = {}
        self.branches: Dict[str, _Branch] = {}
        self.entry: Optional[str] = None

    def add_node(self, name: str, fn: Callable[[State], Optional[State]], *, retries: int = 0,
                 retry_on: Iterable[Type[BaseException]] = (Exception,),
                 backoff: float = 0.0) -> "StateGraph":
        if name in (START, END):
            raise GraphError(f"'{name}' is a reserved node name")
        if name in self.nodes:
            raise GraphError(f"node '{name}' already exists")
        if retries < 0:
            raise GraphError("retries must be >= 0")
        self.nodes[name] = _Node(name, fn, retries, tuple(retry_on), backoff)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        if src == START:
            return self.set_entry(dst)
        if src in self.edges or src in self.branches:
            raise GraphError(f"node '{src}' already has an outgoing edge")
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Callable[[State], str],
                              mapping: Optional[Dict[str, str]] = None) -> "StateGraph":
        if src in self.edges or src in self.branches:
            raise GraphError(f"node '{src}' already has an outgoing edge")
        self.branches[src] = _Branch(router, dict(mapping) if mapping is not None else None)
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    # ------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.entry is None:
            raise GraphError("no entry node: call set_entry() or add_edge(START, ...)")
        known = set(self.nodes) | {END}
        if self.entry not in self.nodes:
            raise GraphError(f"entry node '{self.entry}' does not exist")
        for src in list(self.edges) + list(self.branches):
            if src not in self.nodes:
                raise GraphError(f"edge from unknown node '{src}'")
        for src, dst in self.edges.items():
            if dst not in known:
                raise GraphError(f"edge '{src}' -> unknown node '{dst}'")
        for src, br in self.branches.items():
            for dst in br.targets() or []:
                if dst not in known:
                    raise GraphError(f"conditional edge '{src}' -> unknown node '{dst}'")
        for name in self.nodes:
            if name not in self.edges and name not in self.branches:
                raise GraphError(f"node '{name}' has no outgoing edge (add one to END if it is last)")
        # reachability (only checkable when every branch declares its targets)
        if all(br.mapping is not None for br in self.branches.values()):
            seen, todo = set(), [self.entry]
            while todo:
                n = todo.pop()
                if n in seen or n == END:
                    continue
                seen.add(n)
                todo.extend([self.edges[n]] if n in self.edges else self.branches[n].targets())
            unreachable = sorted(set(self.nodes) - seen)
            if unreachable:
                raise GraphError(f"unreachable node(s): {', '.join(unreachable)}")

    def compile(self, *, max_steps: int = 25, checkpointer: Any = None,
                interrupt_before: Iterable[str] = ()) -> "CompiledGraph":
        self.validate()
        interrupt_before = list(interrupt_before)
        for n in interrupt_before:
            if n not in self.nodes:
                raise GraphError(f"interrupt_before: unknown node '{n}'")
        if max_steps < 1:
            raise GraphError("max_steps must be >= 1")
        return CompiledGraph(self, max_steps, checkpointer, set(interrupt_before))

    # ------------------------------------------------------------ export
    def to_mermaid(self) -> str:
        lines = ["flowchart TD", f"    {START}([start]) --> {self.entry}"]
        for name in self.nodes:
            if name in self.edges:
                lines.append(f"    {name} --> {_m(self.edges[name])}")
            elif self.branches[name].mapping is not None:
                for label, dst in self.branches[name].mapping.items():
                    lines.append(f"    {name} -. {label} .-> {_m(dst)}")
            else:
                lines.append(f"    {name} -. router .-> any")
        return "\n".join(lines)


def _m(node: str) -> str:
    return f"{END}([end])" if node == END else node


# ---------------------------------------------------------------- runner
class CompiledGraph:
    def __init__(self, graph: StateGraph, max_steps: int, checkpointer: Any,
                 interrupt_before: set):
        self.graph = graph
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before

    # public API -------------------------------------------------------
    def invoke(self, state: Optional[State] = None, *, thread_id: Optional[str] = None) -> State:
        return self.run(state, thread_id=thread_id).state

    def run(self, state: Optional[State] = None, *, thread_id: Optional[str] = None) -> RunResult:
        steps = []
        gen = self._execute(state, thread_id, resume=False)
        return self._drain(gen, steps, thread_id)

    def resume(self, thread_id: str, *, update: Optional[State] = None) -> RunResult:
        """Continue a saved run (after an interrupt or a crash). ``update`` is merged into the
        saved state first, e.g. a human's approval."""
        if self.checkpointer is None:
            raise GraphError("resume() needs a checkpointer")
        gen = self._execute(update, thread_id, resume=True)
        return self._drain(gen, [], thread_id)

    def stream(self, state: Optional[State] = None, *,
               thread_id: Optional[str] = None) -> Iterator[Step]:
        gen = self._execute(state, thread_id, resume=False)
        try:
            while True:
                yield next(gen)
        except StopIteration:
            return

    # internals --------------------------------------------------------
    def _drain(self, gen, steps, thread_id) -> RunResult:
        while True:
            try:
                steps.append(next(gen))
            except StopIteration as stop:
                state, status, nxt = stop.value
                return RunResult(state, status, nxt, steps, thread_id)

    def _merge(self, state: State, update: Optional[State]) -> State:
        if not update:
            return state
        if not isinstance(update, dict):
            raise TypeError(f"node must return a dict or None, got {type(update).__name__}")
        new = dict(state)
        for k, v in update.items():
            new[k] = self.graph.reducers.get(k, overwrite)(state.get(k), v)
        return new

    def _next(self, node: str, state: State) -> str:
        if node in self.graph.edges:
            return self.graph.edges[node]
        br = self.graph.branches[node]
        key = br.router(copy.deepcopy(state))
        if br.mapping is not None:
            if key not in br.mapping:
                raise GraphError(f"router of '{node}' returned '{key}', expected one of "
                                 f"{sorted(br.mapping)}")
            return br.mapping[key]
        if key != END and key not in self.graph.nodes:
            raise GraphError(f"router of '{node}' returned unknown node '{key}'")
        return key

    def _call(self, node: _Node, state: State):
        attempts = 0
        while True:
            attempts += 1
            try:
                return node.fn(copy.deepcopy(state)), attempts
            except node.retry_on as exc:  # type: ignore[misc]
                if attempts > node.retries:
                    raise NodeError(node.name, attempts, exc) from exc
                if node.backoff:
                    time.sleep(node.backoff * (2 ** (attempts - 1)))

    def _save(self, thread_id, step, nxt, state, status):
        if self.checkpointer is not None and thread_id is not None:
            self.checkpointer.put(thread_id, {"step": step, "next_node": nxt,
                                              "state": state, "status": status})

    def _execute(self, state, thread_id, resume):
        if resume:
            saved = self.checkpointer.get(thread_id)
            if saved is None:
                raise GraphError(f"no checkpoint for thread '{thread_id}'")
            if saved["status"] == "done":
                return saved["state"], "done", None
            current = self._merge(saved["state"], state)
            node_name, step = saved["next_node"], saved["step"]
            skip_interrupt = True  # the human already reviewed this stop
        else:
            current = dict(state or {})
            node_name, step = self.graph.entry, 0
            skip_interrupt = False

        while node_name != END:
            if node_name in self.interrupt_before and not skip_interrupt:
                self._save(thread_id, step, node_name, current, "interrupted")
                return current, "interrupted", node_name
            skip_interrupt = False
            if step >= self.max_steps:
                raise GraphRecursionError(
                    f"stopped after {self.max_steps} steps (next node '{node_name}'); "
                    "raise max_steps or add an exit condition")
            node = self.graph.nodes[node_name]
            t0 = time.perf_counter()
            update, attempts = self._call(node, current)
            current = self._merge(current, update)
            nxt = self._next(node_name, current)
            step += 1
            self._save(thread_id, step, nxt, current, "running" if nxt != END else "done")
            yield Step(step, node_name, dict(update or {}), nxt, copy.deepcopy(current),
                       attempts, time.perf_counter() - t0)
            node_name = nxt
        return current, "done", None
