import importlib.util
import re

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_the_compiled_graph():
    embedded = render_graph.readme_flow((ROOT / "README.md").read_text())
    assert embedded is not None, "README.md is missing the flow-diagram markers"
    assert embedded == render_graph.render_flow(), "stale diagram: run `python scripts/render_graph.py`"


def test_flow_diagram_names_every_graph_node():
    flow = render_graph.render_flow()
    for node in ("retrieve", "answer_from_kb", "weak_retrieval", "search", "answer_from_search", "finalize"):
        assert re.search(rf"^\s+{node}\(", flow, re.M)


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
