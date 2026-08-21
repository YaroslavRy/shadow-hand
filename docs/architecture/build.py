"""Architecture viewer generator — project-agnostic, tree-structured.

Produces a self-contained `viewer.html` (Ghost-in-the-Shell-themed,
force-directed) that models a project as a single tree:

    root package
      └─ sub-package
           └─ module.py
                └─ class / function / constant   ← always the leaf layer

There is no fixed number of layers. You drill from the top (packages)
down through however many directory levels exist, into a module, and
finally into its code symbols. The deepest layer is always the actual
code; clicking a symbol opens the file at its line.

Modes
-----
  Hand-curated:  reads `model.py` next to this script (or via `--config`).
                 Code symbols are appended automatically by AST scan.
  Auto-discover: `--src DIR` (repeatable). The tree mirrors the directory
                 structure; modules and their imports come from the AST.

Examples
--------
  python build.py --open
  python build.py --src shadow_hand --open
  python build.py --root /other/repo --src . --open      # flat layout
  python build.py --vendor-d3                            # cache d3 offline
"""
from __future__ import annotations

import argparse
import ast
import datetime
import importlib
import importlib.util
import json
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "viewer_template.html"
VENDOR_DIR = HERE / "vendor"
D3_CDN_URL = "https://unpkg.com/d3@7/dist/d3.min.js"


def _short_doc(doc: str | None, limit: int = 64) -> str:
    """First sentence of a docstring, trimmed to `limit` chars."""
    if not doc:
        return ""
    first = doc.strip().split("\n", 1)[0]
    if ". " in first:
        first = first.split(". ", 1)[0] + "."
    first = first.strip()
    if len(first) > limit:
        first = first[: limit - 1].rstrip() + "…"
    return first


# ==========================================================================
# Schema — a single tree. Hierarchy lives entirely in `parent`.
# ==========================================================================
@dataclass(frozen=True)
class Node:
    id: str
    label: str
    parent: str | None = None            # None == top-level
    kind: str = "module"                 # package | module | thread | process
                                         # | external | asset | function
                                         # | class | constant
    file: str | None = None              # repo-relative path
    symbol: str | None = None            # for code-level nodes
    line: int | None = None              # source line for code-level nodes


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    root: Path = field(default_factory=Path.cwd)
    project_name: str = "project"

    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def children_of(self, parent_id: str | None) -> list[Node]:
        return [n for n in self.nodes if n.parent == parent_id]

    def depth_of(self, node_id: str) -> int:
        bi = self.by_id()
        d, cur = 0, bi.get(node_id)
        seen = set()
        while cur and cur.parent and cur.id not in seen:
            seen.add(cur.id)
            cur = bi.get(cur.parent)
            d += 1
        return d

    def max_depth(self) -> int:
        return max((self.depth_of(n.id) for n in self.nodes), default=0)


# ==========================================================================
# Hand-curated config loader
# ==========================================================================
def load_config(path: Path) -> Graph | None:
    spec = importlib.util.spec_from_file_location("_arch_config", path)
    if not (spec and spec.loader):
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_arch_config"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "NODES"):
        return None

    def _coerce_node(n) -> Node:
        d = n.__dict__ if hasattr(n, "__dict__") else n
        return Node(
            id=d["id"], label=d["label"], parent=d.get("parent"),
            kind=d.get("kind", "module"), file=d.get("file"),
            symbol=d.get("symbol"), line=d.get("line"),
        )

    def _coerce_edge(e) -> Edge:
        d = e.__dict__ if hasattr(e, "__dict__") else e
        return Edge(src=d["src"], dst=d["dst"], label=d.get("label", ""))

    return Graph(
        nodes=[_coerce_node(n) for n in mod.NODES],
        edges=[_coerce_edge(e) for e in getattr(mod, "EDGES", [])],
        root=Path(getattr(mod, "PROJECT_ROOT", HERE.parent.parent)).resolve(),
        project_name=getattr(mod, "PROJECT_NAME", path.parent.parent.parent.name),
    )


# ==========================================================================
# AST scan — directory tree + module imports
# ==========================================================================
_THREAD_HINTS = ("threading", "Thread(", "concurrent.futures")
_PROCESS_HINTS = ("multiprocessing", "subprocess.")

_EXCLUDE_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".cache", ".idea", ".vscode",
    "venv", ".venv", "env", ".env", "node_modules",
    "build", "dist", "site-packages", ".archviewer", ".eggs",
    "vendor", "third_party",
}


def _infer_kind(text: str) -> str:
    if any(h in text for h in _PROCESS_HINTS):
        return "process"
    if any(h in text for h in _THREAD_HINTS):
        return "thread"
    return "module"


def _module_name(root: Path, py_file: Path) -> str:
    rel = py_file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else py_file.stem


def _resolve_relative(current_module: str, level: int, name: str | None) -> str:
    parts = current_module.split(".")[:-level] if level else current_module.split(".")
    if name:
        parts.append(name)
    return ".".join(p for p in parts if p)


def scan_project(root: Path, src_dirs: list[str]) -> Graph:
    root = root.resolve()

    # --- pass 1: collect .py files (excluding noise dirs) ------------------
    py_files: list[tuple[Path, Path]] = []  # (file, src_root)
    for sd in src_dirs:
        src = (root / sd).resolve()
        if not src.exists():
            raise FileNotFoundError(f"--src directory not found: {src}")
        for py in src.rglob("*.py"):
            parts = py.relative_to(root).parts
            if any(p in _EXCLUDE_DIRS or p.startswith(".") for p in parts[:-1]):
                continue
            py_files.append((py, src))

    module_by_modname = {
        _module_name(root, py): py.relative_to(root).as_posix()
        for py, _ in py_files
        if py.name != "__init__.py"
    }

    nodes: list[Node] = []
    edges: list[Edge] = []
    seen_pkg: set[str] = set()

    def pkg_id(path: Path) -> str:
        rel = path.relative_to(root).as_posix()
        return rel if rel != "." else root.name

    def add_pkg(path: Path, parent_id: str | None) -> str:
        pid = pkg_id(path)
        if pid in seen_pkg:
            return pid
        seen_pkg.add(pid)
        rel = path.relative_to(root).as_posix()
        nodes.append(Node(
            id=pid,
            label=(path.name + "/") if path != root else (root.name + "/"),
            parent=parent_id, kind="package",
            file=rel if rel != "." else ".",
        ))
        return pid

    # --- pass 2: build the tree -------------------------------------------
    for py, src in py_files:
        if py.name == "__init__.py":
            # __init__.py only contributes its package node, not a module.
            add_pkg(py.parent, _parent_pkg_id(py.parent, src, root, add_pkg))
            continue

        leaf_pkg = _parent_pkg_id(py.parent, src, root, add_pkg)

        text = py.read_text(errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        kind = _infer_kind(text)
        doc = _short_doc(ast.get_docstring(tree))
        mod_id = py.relative_to(root).as_posix()
        nodes.append(Node(
            id=mod_id, label=py.name, parent=leaf_pkg,
            kind=kind, file=mod_id,
        ))

        targets: set[str] = set()
        modname = _module_name(root, py)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                targets.add(_resolve_relative(modname, n.level or 0, n.module))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    targets.add(a.name)
        for tgt in targets:
            cur = tgt
            while cur:
                if cur in module_by_modname and module_by_modname[cur] != mod_id:
                    edges.append(Edge(src=mod_id, dst=module_by_modname[cur]))
                    break
                cur = cur.rsplit(".", 1)[0] if "." in cur else ""

    # dedupe import edges
    seen_e: set[tuple[str, str]] = set()
    dedup: list[Edge] = []
    for e in edges:
        if (e.src, e.dst) in seen_e:
            continue
        seen_e.add((e.src, e.dst))
        dedup.append(e)

    return Graph(nodes=nodes, edges=dedup, root=root, project_name=root.name)


def _parent_pkg_id(dir_path: Path, src: Path, root: Path, add_pkg) -> str:
    """Ensure the package chain from `src` down to `dir_path` exists.
    Returns the id of `dir_path`'s package node."""
    # Build the ancestor chain src .. dir_path inclusive.
    chain: list[Path] = []
    cur = dir_path
    while True:
        chain.append(cur)
        if cur == src:
            break
        cur = cur.parent
    chain.reverse()  # src first

    parent_id: str | None = None
    for d in chain:
        parent_id = add_pkg(d, parent_id)
    return parent_id  # type: ignore[return-value]


# ==========================================================================
# AST scan — code symbols (leaf layer), appended under each module node
# ==========================================================================
def _scan_module_symbols(
    file_path: Path, parent_id: str, root: Path,
) -> tuple[list[Node], list[Edge]]:
    try:
        text = file_path.read_text(errors="replace")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return [], []

    rel = file_path.relative_to(root).as_posix()
    nodes: list[Node] = []
    edges: list[Edge] = []
    symbols: dict[str, str] = {}

    def _add(name: str, kind: str, lineno: int, label: str) -> str:
        sid = f"{parent_id}::{name}"
        symbols[name] = sid
        nodes.append(Node(
            id=sid, label=label, parent=parent_id,
            kind=kind, file=rel, symbol=name, line=lineno,
        ))
        return sid

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = _short_doc(ast.get_docstring(n))
            _add(n.name, "function", n.lineno, f"{n.name}()\n{doc or (rel + ':' + str(n.lineno))}")
        elif isinstance(n, ast.ClassDef):
            methods = sum(1 for c in n.body
                          if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)))
            doc = _short_doc(ast.get_docstring(n))
            parts = [f"{methods} methods"]
            if doc:
                parts.append(doc)
            _add(n.name, "class", n.lineno, f"class {n.name}\n{' / '.join(parts)}")
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if (isinstance(t, ast.Name) and t.id.isupper()
                        and not t.id.startswith("_") and len(t.id) > 1):
                    _add(t.id, "constant", n.lineno, f"{t.id}\n{rel}:{n.lineno}")

    for top in tree.body:
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        src_id = symbols.get(top.name)
        if not src_id:
            continue
        for sub in ast.walk(top):
            if isinstance(sub, ast.Call):
                callee = None
                if isinstance(sub.func, ast.Name):
                    callee = sub.func.id
                elif isinstance(sub.func, ast.Attribute):
                    callee = sub.func.attr
                if callee and callee in symbols and symbols[callee] != src_id:
                    edges.append(Edge(src=src_id, dst=symbols[callee]))
    return nodes, edges


def augment_code_level(graph: Graph) -> None:
    """Append code-symbol leaves under every Python-module node."""
    extra_nodes: list[Node] = []
    extra_edges: list[Edge] = []
    for n in list(graph.nodes):
        if n.kind not in ("module", "thread", "process"):
            continue
        if not n.file or not n.file.endswith(".py"):
            continue
        path = graph.root / n.file
        if not path.exists():
            continue
        sn, se = _scan_module_symbols(path, parent_id=n.id, root=graph.root)
        extra_nodes.extend(sn)
        extra_edges.extend(se)
    graph.nodes.extend(extra_nodes)
    graph.edges.extend(extra_edges)


# ==========================================================================
# Validation + stats
# ==========================================================================
def validate(graph: Graph) -> list[str]:
    out: list[str] = []
    seen = {n.id for n in graph.nodes}
    assert len(seen) == len(graph.nodes), "duplicate node ids"
    for n in graph.nodes:
        if n.parent is not None:
            assert n.parent in seen, f"node {n.id}: parent {n.parent!r} missing"
    for e in graph.edges:
        assert e.src in seen, f"edge {e.src!r} -> {e.dst!r}: src missing"
        assert e.dst in seen, f"edge {e.src!r} -> {e.dst!r}: dst missing"

    # No cycles in the parent chain.
    bi = graph.by_id()
    for n in graph.nodes:
        seen_chain, cur = set(), n
        while cur and cur.parent:
            assert cur.id not in seen_chain, f"cycle in parent chain at {n.id}"
            seen_chain.add(cur.id)
            cur = bi.get(cur.parent)

    for n in graph.nodes:
        if n.symbol is None or n.file is None:
            continue
        if n.kind in ("function", "class", "constant"):
            continue  # AST-derived, trusted
        path = graph.root / n.file
        if not path.exists():
            out.append(f"  [miss ] {n.id}  ({n.file})")
            continue
        mod_name = ".".join(Path(n.file).with_suffix("").parts)
        if str(graph.root) not in sys.path:
            sys.path.insert(0, str(graph.root))
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            out.append(f"  [defer] {n.id}.{n.symbol}")
            continue
        if hasattr(mod, n.symbol):
            out.append(f"  [check] {n.id}.{n.symbol}")
        else:
            out.append(f"  [WARN ] {n.id}: {n.file} lacks `{n.symbol}`")
    return out


def compute_stats(graph: Graph) -> dict:
    kinds: dict[str, int] = {}
    for n in graph.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    files = {n.file for n in graph.nodes if n.file and n.file.endswith(".py")}
    depths = [graph.depth_of(n.id) for n in graph.nodes]
    return {
        "nodes_total": len(graph.nodes),
        "edges_total": len(graph.edges),
        "indexed_files": len(files),
        "max_depth": max(depths, default=0),
        "kinds": kinds,
        "packages": kinds.get("package", 0),
        "modules": kinds.get("module", 0) + kinds.get("thread", 0) + kinds.get("process", 0),
        "symbols": kinds.get("function", 0) + kinds.get("class", 0) + kinds.get("constant", 0),
    }


# ==========================================================================
# Build
# ==========================================================================
def _graph_data(graph: Graph) -> dict:
    def node_to_json(n: Node) -> dict:
        title, *rest = n.label.split("\n", 1)
        body = rest[0] if rest else ""
        link = None
        if n.file and not Path(n.file).is_dir() and n.file != ".":
            target = (graph.root / n.file).resolve()
            link = f"vscode://file/{target}"
            if n.line is not None:
                link += f":{n.line}"
        return {
            "id": n.id, "parent": n.parent, "kind": n.kind,
            "title": title, "body": body,
            "file": n.file, "symbol": n.symbol, "line": n.line, "link": link,
        }
    return {
        "nodes": [node_to_json(n) for n in graph.nodes],
        "edges": [{"source": e.src, "target": e.dst, "label": e.label}
                  for e in graph.edges],
    }


def _vendor_d3_path() -> Path:
    return VENDOR_DIR / "d3.min.js"


def fetch_d3(force: bool = False) -> Path:
    target = _vendor_d3_path()
    if target.exists() and not force:
        return target
    import urllib.request
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"// fetching {D3_CDN_URL} -> {target}")
    urllib.request.urlretrieve(D3_CDN_URL, target)
    return target


def _d3_script_tag() -> str:
    if _vendor_d3_path().exists():
        content = _vendor_d3_path().read_text()
        return f"<!-- d3@7 inlined from vendor/d3.min.js -->\n<script>{content}</script>"
    return f'<script src="{D3_CDN_URL}"></script>'


def build(graph: Graph, out: Path) -> Path:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"viewer_template.html not found at {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = (template
            .replace("__DATA__", json.dumps(_graph_data(graph)))
            .replace("__STATS__", json.dumps(compute_stats(graph)))
            .replace("__PROJECT__", graph.project_name)
            .replace("__BUILD_STAMP__", stamp)
            .replace("__D3_SCRIPT__", _d3_script_tag()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


# ==========================================================================
# CLI
# ==========================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=None, type=Path)
    p.add_argument("--src", action="append", default=[], metavar="DIR")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=HERE / "viewer.html")
    p.add_argument("--open", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--no-code-level", action="store_true")
    p.add_argument("--vendor-d3", action="store_true",
                   help="Download d3.min.js to vendor/ for offline builds.")
    args = p.parse_args()

    if args.vendor_d3:
        fetch_d3(force=True)

    t0 = time.perf_counter()
    graph: Graph | None = None
    if args.config:
        graph = load_config(args.config)
        if graph is None:
            sys.exit(f"--config {args.config} did not yield a NODES list")
    elif (HERE / "model.py").exists() and not args.src and not args.root:
        # Only fall back to the local curated model when the user hasn't
        # pointed us at another project (--root) or chosen dirs (--src).
        graph = load_config(HERE / "model.py")

    if graph is None:
        root = (args.root or Path.cwd()).resolve()
        # Default to scanning the whole root when no --src is given.
        srcs = args.src or ["."]
        graph = scan_project(root, srcs)
    elif args.root:
        # Curated model + explicit --root: trust the model's own root,
        # don't blindly relocate it (that breaks file resolution).
        graph.root = args.root.resolve()

    if not args.no_code_level:
        augment_code_level(graph)
    scan_ms = (time.perf_counter() - t0) * 1000

    s = compute_stats(graph)
    print(f"// project: {graph.project_name}")
    print(f"// root:    {graph.root}")
    print(f"// scan:    {scan_ms:.1f} ms")
    print(f"// tree:    {s['packages']} packages · {s['modules']} modules · "
          f"{s['symbols']} symbols · max depth {s['max_depth']}")
    print(f"// totals:  {s['nodes_total']} nodes · {s['edges_total']} edges · "
          f"{s['indexed_files']} files")

    if not args.no_validate:
        print("// validating...")
        for line in validate(graph):
            print(line)

    out = build(graph, args.out)
    print(f"// wrote {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
