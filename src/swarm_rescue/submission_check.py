import ast
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

SUBMISSION_ZIP_RE = re.compile(
    r"^team(\d{3})_(preliminary|semifinal|final)\.zip$"
)

EVAL_STEPS = ("preliminary", "semifinal", "final")

REQUIRED_ZIP_ENTRIES = (
    "my_drone_eval.py",
    "team_info.yml",
    "req.txt",
)

FORBIDDEN_ZIP_MARKERS = (
    ".venv/",
    "swarm_rescue/",
    "__pycache__/",
    ".git/",
    "node_modules/",
)

FORBIDDEN_ZIP_PREFIXES = (
    "maps/",
    "simulation/",
    "config/",
)

FORBIDDEN_ZIP_BASENAMES = frozenset({
    "launcher.py",
})

# Bundled weights are allowed for deep-learning submissions (see submission_guidelines.md).
MODEL_WEIGHT_SUFFIXES = (
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".h5",
    ".pb",
    ".safetensors",
    ".pkl",
    ".pickle",
)

ZIP_SIZE_SOFT_WARN_BYTES = 500 * 1024
ZIP_SIZE_SOFT_WARN_WITH_MODEL_BYTES = 30 * 1024 * 1024
ZIP_SIZE_WARN_BYTES = 100 * 1024 * 1024
LARGE_FILE_WARN_BYTES = 10 * 1024 * 1024

TORCH_REQ_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})

DEPRECATED_API_NAMES = frozenset({
    "WoundedPerson",
    "RescueCenter",
})

DEPRECATED_API_ATTRS = frozenset({
    "WOUNDED_PERSON",
    "RESCUE_CENTER",
    "grasped_wounded_persons",
})

DEPRECATED_API_SUBSTRINGS = (
    "number_wounded_persons",
    '"type": "rescue_center"',
    "'type': 'rescue_center'",
)

FORBIDDEN_TRUE_CALLS = frozenset({
    "true_position",
    "true_angle",
    "true_velocity",
    "true_angular_velocity",
})

DISABLED_SELF_ATTRS = frozenset({
    "position",
    "angle",
    "velocity",
    "angular_velocity",
})

DEBUG_ATTR_CALLS = frozenset({
    "imshow",
    "waitKey",
})

WORKSPACE_SOLUTIONS = Path("src/swarm_rescue/solutions")
WORKSPACE_REQ = Path("req.txt")


@dataclass
class SubmissionCheckResult:
    """Aggregated submission validation outcome."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def is_valid_submission_name(name: str) -> bool:
    return SUBMISSION_ZIP_RE.match(name) is not None


def parse_submission_zip_name(name: str) -> Optional[Tuple[int, str]]:
    """Return (team_number, evalstep) when name is valid."""
    match = SUBMISSION_ZIP_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def list_submission_zips(directory: Path) -> List[Path]:
    """Return sorted valid submission zip paths in directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Submissions directory not found: {directory}")

    zips = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and is_valid_submission_name(p.name)
    ]
    return zips


def zip_stem(zip_path: Path) -> str:
    return zip_path.stem


def validate_zip_contents(zip_path: Path) -> Tuple[bool, Optional[str]]:
    """Check that zip contains required submission files."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False, f"Invalid zip file: {zip_path}"

    normalized = set()
    for name in names:
        base = name.split("/")[-1] if "/" in name else name
        if base:
            normalized.add(base)

    missing = [req for req in REQUIRED_ZIP_ENTRIES if req not in normalized]
    if missing:
        return False, f"Missing required files in {zip_path.name}: {', '.join(missing)}"
    return True, None


def _collect_zip_names(zip_path: Path) -> Tuple[Optional[set], Optional[str]]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return set(zf.namelist()), None
    except zipfile.BadZipFile:
        return None, f"Invalid zip file: {zip_path}"


def validate_team_info_yaml(
    yaml_path: Path,
    *,
    expected_team_number: Optional[int] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """Parse team_info.yml and validate required fields."""
    if not yaml_path.is_file():
        return None, f"team_info.yml not found: {yaml_path}"

    try:
        raw = yaml_path.read_text(encoding="utf-8")
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML in team_info.yml: {exc}"

    if not isinstance(config, dict):
        return None, "team_info.yml must contain a YAML mapping"

    for key in ("team_number", "team_name", "team_members"):
        if key not in config or config[key] in (None, ""):
            return None, f"team_info.yml: missing or empty field '{key}'"

    try:
        team_number = int(config["team_number"])
    except (TypeError, ValueError):
        return None, "team_info.yml: team_number must be an integer"

    if team_number < 0 or team_number > 999:
        return None, "team_info.yml: team_number must be between 0 and 999"

    team_name = str(config["team_name"]).strip()
    team_members = str(config["team_members"]).strip()
    if not team_name:
        return None, "team_info.yml: team_name must not be empty"
    if not team_members:
        return None, "team_info.yml: team_members must not be empty"

    if expected_team_number is not None and team_number != expected_team_number:
        return None, (
            f"team_info.yml team_number ({team_number}) does not match "
            f"zip file team number ({expected_team_number:03d})"
        )

    return {
        "team_number": team_number,
        "team_name": team_name,
        "team_members": team_members,
    }, None


def validate_my_drone_eval_source(source_path: Path) -> Optional[str]:
    """Syntax-check my_drone_eval.py and verify an entry point is defined."""
    if not source_path.is_file():
        return f"my_drone_eval.py not found: {source_path}"

    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        return f"my_drone_eval.py syntax error: {exc}"

    has_drone_class_for_mode = False
    has_my_drone_eval_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "drone_class_for_mode":
            has_drone_class_for_mode = True
        if isinstance(node, ast.ClassDef) and node.name == "MyDroneEval":
            has_my_drone_eval_class = True

    if not has_drone_class_for_mode and not has_my_drone_eval_class:
        return (
            "my_drone_eval.py must define drone_class_for_mode(mode) "
            "or class MyDroneEval"
        )
    return None


def validate_req_txt(req_path: Path) -> Optional[str]:
    if not req_path.is_file():
        return f"req.txt not found: {req_path}"
    return None


def validate_req_txt_content(req_path: Path) -> SubmissionCheckResult:
    """Validate req.txt format (one dependency per line)."""
    result = SubmissionCheckResult()
    if not req_path.is_file():
        result.add_error(f"req.txt not found: {req_path}")
        return result

    text = req_path.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            result.add_warning(
                f"req.txt line {line_no}: pip flags are unusual in req.txt ({line!r})"
            )
            continue
        if "://" in line or "/" in line or "\\" in line:
            result.add_error(
                f"req.txt line {line_no}: use package names only, not paths or URLs ({line!r})"
            )
            continue
        if len(line.split()) > 1 and not line.startswith("-e "):
            result.add_error(
                f"req.txt line {line_no}: one package per line ({line!r})"
            )
            continue
        package = line.split("[", 1)[0].strip().lower()
        base_pkg = package.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0]
        if base_pkg in TORCH_REQ_PACKAGES:
            result.add_warning(
                f"req.txt line {line_no}: pinning {base_pkg} overrides the evaluator "
                "CUDA PyTorch stack and may cause incompatibilities"
            )
    return result


def _python_sources_from_directory(solutions_dir: Path) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = []
    for path in sorted(solutions_dir.rglob("*.py")):
        rel = str(path.relative_to(solutions_dir))
        sources.append((rel, path.read_text(encoding="utf-8")))
    return sources


def _python_sources_from_zip(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = []
    for name in zf.namelist():
        norm = name.replace("\\", "/")
        if not norm.endswith(".py") or norm.endswith("/"):
            continue
        try:
            content = zf.read(name).decode("utf-8")
        except UnicodeDecodeError:
            content = zf.read(name).decode("utf-8", errors="replace")
        rel = norm.split("/")[-1] if "/" in norm else norm
        sources.append((norm, content))
    return sources


def _enclosing_function_name(tree: ast.AST, lineno: int) -> str:
    innermost: Optional[Tuple[int, str]] = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= lineno <= end:
            if innermost is None or node.lineno >= innermost[0]:
                innermost = (node.lineno, node.name)
    if innermost is None:
        return "<module level>"
    return innermost[1]


def _check_ground_truth_calls(tree: ast.AST, rel_path: str, result: SubmissionCheckResult) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in FORBIDDEN_TRUE_CALLS:
            continue
        where = _enclosing_function_name(tree, node.lineno)
        result.add_error(
            f"{rel_path}:{node.lineno}: {where} calls {node.func.attr}() "
            "(ground-truth APIs are disabled during evaluation; use measured sensors)"
        )


def _check_control_bodies(tree: ast.AST, rel_path: str, result: SubmissionCheckResult) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "control":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr in DISABLED_SELF_ATTRS
            ):
                result.add_error(
                    f"{rel_path}: control() uses self.{child.attr} "
                    "(disabled; use measured_* API)"
                )


def _check_deprecated_api(tree: ast.AST, rel_path: str, result: SubmissionCheckResult) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DEPRECATED_API_NAMES:
            result.add_error(
                f"{rel_path}: uses deprecated symbol {node.id} "
                "(see README.md API naming table)"
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name in DEPRECATED_API_NAMES:
                    result.add_error(
                        f"{rel_path}: uses deprecated symbol {alias.name} "
                        "(see README.md API naming table)"
                    )
        if isinstance(node, ast.Attribute) and node.attr in DEPRECATED_API_ATTRS:
            result.add_error(
                f"{rel_path}: uses deprecated API .{node.attr} "
                "(see README.md API naming table)"
            )


def _check_debug_calls(tree: ast.AST, rel_path: str, result: SubmissionCheckResult) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in DEBUG_ATTR_CALLS:
                result.add_warning(
                    f"{rel_path}: possible debug UI call .{node.func.attr}() "
                    "(disable before submission)"
                )


def scan_submission_python_sources(
    sources: List[Tuple[str, str]],
    result: SubmissionCheckResult,
) -> None:
    """AST checks on all Python files included in the submission."""
    uses_torch = False
    print_count = 0

    for rel_path, source in sources:
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            result.add_error(f"{rel_path}: syntax error: {exc}")
            continue

        _check_deprecated_api(tree, rel_path, result)
        _check_ground_truth_calls(tree, rel_path, result)
        _check_control_bodies(tree, rel_path, result)
        _check_debug_calls(tree, rel_path, result)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        uses_torch = True
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "torch" or node.module.startswith("torch."):
                    uses_torch = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    print_count += 1

        for marker in DEPRECATED_API_SUBSTRINGS:
            if marker in source:
                result.add_error(
                    f"{rel_path}: contains deprecated pattern {marker!r} "
                    "(see README.md API naming table)"
                )

    if print_count > 15:
        result.add_warning(
            f"Found {print_count} print() calls in submission code; "
            "reduce console output before submitting"
        )
    if uses_torch:
        result.add_warning(
            "Submission imports torch; ensure GPU eval runs with --gpus all "
            "and avoid pinning conflicting torch* versions in req.txt"
        )


def _zip_has_model_weights(names: set) -> bool:
    for entry in names:
        lower = entry.replace("\\", "/").lower()
        if any(lower.endswith(suffix) for suffix in MODEL_WEIGHT_SUFFIXES):
            return True
    return False


def _check_zip_entries(names: set, result: SubmissionCheckResult, zip_path: Path) -> None:
    for entry in names:
        normalized = entry.replace("\\", "/")
        if normalized.endswith("/"):
            continue
        base = normalized.rsplit("/", 1)[-1]

        for prefix in FORBIDDEN_ZIP_PREFIXES:
            if normalized.startswith(prefix) or f"/{prefix}" in normalized:
                result.add_error(
                    f"Zip must not include project {prefix.rstrip('/')} "
                    f"(found entry: {entry})"
                )
                break

        if base in FORBIDDEN_ZIP_BASENAMES:
            result.add_error(
                f"Zip must not include {base} (submit solutions/ content only)"
            )

        for marker in FORBIDDEN_ZIP_MARKERS:
            if marker in normalized:
                result.add_error(
                    f"Zip must not contain '{marker.rstrip('/')}' "
                    f"(found entry: {entry})"
                )
                break

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.file_size <= LARGE_FILE_WARN_BYTES:
                    continue
                norm = info.filename.replace("\\", "/").lower()
                is_weight = any(norm.endswith(suffix) for suffix in MODEL_WEIGHT_SUFFIXES)
                if is_weight:
                    continue
                result.add_warning(
                    f"Large non-model file in zip ({info.filename}, "
                    f"{info.file_size // 1024} KiB); check that .venv was not included"
                )
    except zipfile.BadZipFile:
        pass


def _apply_submission_content_checks(
    result: SubmissionCheckResult,
    *,
    req_path: Optional[Path],
    python_sources: List[Tuple[str, str]],
) -> None:
    if req_path is not None:
        req_result = validate_req_txt_content(req_path)
        result.errors.extend(req_result.errors)
        result.warnings.extend(req_result.warnings)
    scan_submission_python_sources(python_sources, result)


def validate_zip_archive(zip_path: Path) -> SubmissionCheckResult:
    """Full validation for a submission zip (evaluator / post-build check)."""
    result = SubmissionCheckResult()
    path = zip_path.resolve()

    if not path.is_file():
        result.add_error(f"File not found: {path}")
        return result

    if not is_valid_submission_name(path.name):
        result.add_error(
            "Zip file name must match teamNNN_{preliminary,semifinal,final}.zip "
            f"(got {path.name})"
        )
        return result

    parsed = parse_submission_zip_name(path.name)
    assert parsed is not None
    team_number, _evalstep = parsed

    ok, err = validate_zip_contents(path)
    if not ok:
        result.add_error(err)
        return result

    names, zip_err = _collect_zip_names(path)
    if zip_err:
        result.add_error(zip_err)
        return result
    assert names is not None

    _check_zip_entries(names, result, path)

    for required in ("my_drone_eval.py", "team_info.yml", "req.txt"):
        if required not in names:
            nested = [
                n for n in names
                if n.replace("\\", "/").endswith(f"/{required}")
                and "/" in n.replace("\\", "/").rstrip("/")
            ]
            if nested:
                result.add_error(
                    f"{required} must be at the zip root, not inside a folder "
                    f"(found: {nested[0]}). Zip the contents of your solutions "
                    "folder, not the folder itself."
                )

    size = path.stat().st_size
    has_models = _zip_has_model_weights(names)
    soft_limit = (
        ZIP_SIZE_SOFT_WARN_WITH_MODEL_BYTES
        if has_models
        else ZIP_SIZE_SOFT_WARN_BYTES
    )
    if size > soft_limit:
        if has_models:
            result.add_warning(
                f"Zip file is {size // 1024} KiB with bundled model weights; "
                "ensure .venv and swarm_rescue are not included"
            )
        else:
            result.add_warning(
                f"Zip file is {size // 1024} KiB (guidelines recommend a few KiB "
                "for code-only submissions)"
            )
    if size > ZIP_SIZE_WARN_BYTES:
        result.add_warning(
            f"Zip file is very large ({size // 1024} KiB). "
            "Exclude .venv and the swarm_rescue package (model weights are allowed)."
        )

    with zipfile.ZipFile(path, "r") as zf:
        python_sources = _python_sources_from_zip(zf)
        tmp_dir = path.parent
        req_member = next(
            (n for n in zf.namelist() if n.replace("\\", "/").endswith("req.txt")),
            None,
        )
        extracted_req: Optional[Path] = None
        if req_member:
            extracted_req = tmp_dir / f".__check_{path.stem}_req.txt"
            extracted_req.write_bytes(zf.read(req_member))
        team_info_member = next(
            (n for n in zf.namelist() if n.replace("\\", "/").endswith("team_info.yml")),
            None,
        )
        my_drone_member = next(
            (n for n in zf.namelist() if n.replace("\\", "/").endswith("my_drone_eval.py")),
            None,
        )
        if team_info_member:
            extracted_team_info = tmp_dir / f".__check_{path.stem}_team_info.yml"
            try:
                extracted_team_info.write_bytes(zf.read(team_info_member))
                _, team_err = validate_team_info_yaml(
                    extracted_team_info,
                    expected_team_number=team_number,
                )
                if team_err:
                    result.add_error(team_err)
            finally:
                extracted_team_info.unlink(missing_ok=True)

        if my_drone_member:
            extracted_drone = tmp_dir / f".__check_{path.stem}_my_drone_eval.py"
            try:
                extracted_drone.write_bytes(zf.read(my_drone_member))
                drone_err = validate_my_drone_eval_source(extracted_drone)
                if drone_err:
                    result.add_error(drone_err)
            finally:
                extracted_drone.unlink(missing_ok=True)

        try:
            _apply_submission_content_checks(
                result,
                req_path=extracted_req,
                python_sources=python_sources,
            )
        finally:
            if extracted_req is not None:
                extracted_req.unlink(missing_ok=True)

    return result


def validate_workspace(root: Path) -> SubmissionCheckResult:
    """Validate a local project tree before creating the submission zip."""
    result = SubmissionCheckResult()
    root = root.resolve()

    req_path = root / WORKSPACE_REQ
    req_err = validate_req_txt(req_path)
    if req_err:
        result.add_error(req_err)
    else:
        req_result = validate_req_txt_content(req_path)
        result.errors.extend(req_result.errors)
        result.warnings.extend(req_result.warnings)

    solutions = root / WORKSPACE_SOLUTIONS
    if not solutions.is_dir():
        result.add_error(f"Solutions directory not found: {solutions}")
        return result

    for forbidden_name in (".venv", "swarm_rescue"):
        forbidden = solutions / forbidden_name
        if forbidden.exists():
            result.add_error(
                f"Remove {forbidden.relative_to(root)} from your submission "
                "(must not be included)"
            )

    team_info_path = solutions / "team_info.yml"
    _, team_err = validate_team_info_yaml(team_info_path)
    if team_err:
        result.add_error(team_err)

    drone_path = solutions / "my_drone_eval.py"
    drone_err = validate_my_drone_eval_source(drone_path)
    if drone_err:
        result.add_error(drone_err)

    scan_submission_python_sources(_python_sources_from_directory(solutions), result)

    return result


def validate_submission(target: Path) -> SubmissionCheckResult:
    """Validate workspace root or a submission zip path."""
    target = target.resolve()
    if target.is_file() and target.suffix.lower() == ".zip":
        return validate_zip_archive(target)
    if target.is_dir():
        if (target / WORKSPACE_REQ).is_file() or (target / WORKSPACE_SOLUTIONS).is_dir():
            return validate_workspace(target)
        zip_candidates = [
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() == ".zip"
        ]
        if len(zip_candidates) == 1:
            return validate_zip_archive(zip_candidates[0])
        if zip_candidates:
            result = SubmissionCheckResult()
            result.add_error(
                f"Directory contains multiple zip files; pass the zip path explicitly: "
                f"{', '.join(p.name for p in sorted(zip_candidates))}"
            )
            return result
        result = SubmissionCheckResult()
        result.add_error(
            f"Not a swarm-rescue workspace or submission zip: {target}"
        )
        return result
    result = SubmissionCheckResult()
    result.add_error(f"Path not found: {target}")
    return result
