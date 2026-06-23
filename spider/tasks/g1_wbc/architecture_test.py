from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SIMULATOR_PATH = REPO_ROOT / "spider" / "simulators" / "g1_wbc.py"
TASK_PATH = REPO_ROOT / "spider" / "tasks" / "g1_wbc" / "spider_task.py"
EVALUATE_PATH = REPO_ROOT / "spider" / "tasks" / "g1_wbc" / "evaluate.py"
VISUALIZE_PATH = REPO_ROOT / "scripts" / "visualize_g1_wbc.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_g1_wbc_simulator_is_backend_only() -> None:
    tree = _parse(SIMULATOR_PATH)
    imports = _imported_modules(tree)
    names = _top_level_names(tree)

    assert "spider.optimizers.receding" not in imports
    assert "spider.optimizers.sampling" not in imports
    assert "spider.tasks.g1_wbc.spider_task" not in imports
    assert "G1WbcSamplingTask" not in names
    assert "G1WbcSpiderResult" not in names
    assert "REWARD_WEIGHT_PRESETS" not in names
    assert "initial_controls" not in names
    assert "tail_controls" not in names
    assert "rollout" not in names
    assert "execute" not in names
    assert "build_result" not in names


def test_g1_wbc_task_adapter_uses_generic_spider_optimizer() -> None:
    tree = _parse(TASK_PATH)
    imports = _imported_modules(tree)

    assert "spider.optimizers.receding" in imports
    assert "spider.optimizers.sampling" in imports
    assert "spider.simulators.g1_wbc" in imports


def test_g1_wbc_cli_uses_task_adapter_not_simulator_task_exports() -> None:
    tree = _parse(EVALUATE_PATH)
    imports = _imported_modules(tree)

    assert "spider.tasks.g1_wbc.spider_task" in imports
    assert "spider.simulators.g1_wbc" not in imports
    assert "spider.optimizers.receding" not in imports
    assert "build_sampling_mpc_config" not in _imported_names(tree)


def test_g1_wbc_visualizer_reuses_task_adapter() -> None:
    tree = _parse(VISUALIZE_PATH)
    imports = _imported_modules(tree)

    assert "spider.tasks.g1_wbc.spider_task" in imports
    assert "spider.optimizers.receding" not in imports
    assert "build_sampling_mpc_config" not in _imported_names(tree)
