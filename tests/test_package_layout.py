from pathlib import Path
import subprocess
import sys

from text_extractor.core.normalize import normalize_scene_output
from text_extractor.core.operations import OperationLog
from text_extractor.core.registry import ModelDefinition, index_definitions, model_options
from text_extractor.core.schema import parse_schema
from text_extractor.validation.strict import EN_TO_CN, canonicalize_fields


def test_shared_modules_import_from_text_extractor():
    public_objects = (
        normalize_scene_output,
        OperationLog,
        ModelDefinition,
        index_definitions,
        model_options,
        parse_schema,
        canonicalize_fields,
    )

    assert all(item.__module__.startswith("text_extractor.") for item in public_objects)
    assert EN_TO_CN["ISO"] == "感光度"


def test_removed_pp_uie_package_is_not_present_or_importable():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; assert importlib.util.find_spec('pp_uie') is None",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert not (project_root / "pp_uie").exists()
    assert result.returncode == 0, result.stderr


def test_backend_family_imports_do_not_load_each_other():
    project_root = Path(__file__).resolve().parents[1]
    checks = (
        ("text_extractor.backends.uie", "text_extractor.backends.pp_uie"),
        ("text_extractor.backends.pp_uie", "text_extractor.backends.uie"),
    )

    for imported, forbidden in checks:
        code = (
            f"import sys; import {imported}; "
            f"assert {forbidden!r} not in sys.modules, sorted(sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
