import importlib.util

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_the_compiled_graph():
    embedded = render_graph.readme_flow((ROOT / "README.md").read_text())
    assert embedded is not None, "README.md is missing the flow-diagram markers"
    assert render_graph.mermaid_shape(embedded) == render_graph.graph_shape(), (
        "stale diagram: run `python scripts/render_graph.py`"
    )


def test_rendered_flow_has_every_graph_node_and_edge():
    nodes, edges = render_graph.graph_shape()
    assert {"retrieve", "answer_from_kb", "weak_retrieval", "search", "answer_from_search", "finalize"} <= nodes
    assert ("answer_from_search", "finalize", False) in edges
    assert ("retrieve", "weak_retrieval", True) in edges
    assert render_graph.mermaid_shape(render_graph.render_flow()) == (nodes, edges)


def test_render_script_check_detects_drift(tmp_path, monkeypatch):
    stale = tmp_path / "README.md"
    stale.write_text(
        "<!-- flow-diagram:start -->\n```mermaid\ngraph TD;\n```\n<!-- flow-diagram:end -->\n"
    )
    monkeypatch.setattr(render_graph, "README", stale)
    monkeypatch.setattr("sys.argv", ["render_graph.py", "--check"])
    assert render_graph.main() == 1
    monkeypatch.setattr("sys.argv", ["render_graph.py"])
    assert render_graph.main() == 0
    monkeypatch.setattr("sys.argv", ["render_graph.py", "--check"])
    assert render_graph.main() == 0


def test_check_ignores_formatting_but_not_edges(tmp_path, monkeypatch):
    flow = render_graph.render_flow()
    reformatted = "\n".join(line.strip() for line in flow.splitlines() if not line.strip().startswith("classDef"))
    readme = tmp_path / "README.md"
    monkeypatch.setattr(render_graph, "README", readme)
    monkeypatch.setattr("sys.argv", ["render_graph.py", "--check"])

    readme.write_text(f"<!-- flow-diagram:start -->\n```mermaid\n{reformatted}\n```\n<!-- flow-diagram:end -->\n")
    assert render_graph.main() == 0

    dropped = reformatted.replace("answer_from_search --> finalize;", "")
    readme.write_text(f"<!-- flow-diagram:start -->\n```mermaid\n{dropped}\n```\n<!-- flow-diagram:end -->\n")
    assert render_graph.main() == 1
