"""Keep the three component dependency directions enforceable."""

import ast
from pathlib import Path


def test_component_import_boundaries():
    root = Path(__file__).resolve().parents[2]
    forbidden = {
        "backend": {"frontend", "training", "PyQt5", "albumentations"},
        "frontend": {"training"},
        "training": {"frontend", "PyQt5"},
    }
    for component, blocked in forbidden.items():
        for path in (root / component).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not {name.split(".")[0] for name in names} & blocked, path
