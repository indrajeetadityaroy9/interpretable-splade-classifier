import glob
import inspect
import os
import re

import pytest

CODE_DIR = "splade"
SCRIPTS_DIR = "splade/scripts"

def get_python_files():
    files = glob.glob(os.path.join(CODE_DIR, "**", "*.py"), recursive=True)
    return [f for f in files if "tests" not in f]

@pytest.mark.parametrize("file_path", get_python_files())
def test_no_os_environ_control(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    if "utils/cuda.py" in file_path:
        return
    matches = re.findall(r'os\.environ\.get\(', content)
    if matches:
        pytest.fail(f"Found os.environ.get usage in {file_path}. Configuration must be via YAML.")

@pytest.mark.parametrize("file_path", glob.glob(os.path.join(SCRIPTS_DIR, "*.py")))
def test_scripts_argparse_clean(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    matches = re.findall(r'\.add_argument\s*\(\s*(["\']--?[\w-]+["\'])', content)

    for arg_name in matches:
        arg_name = arg_name.strip("'\"")
        if arg_name not in ["--config", "--help"]:
             pytest.fail(f"Found extra CLI argument {arg_name} in {file_path}. Only --config is allowed.")


def test_cis_hardwired():
    """Verify CIS mechanisms are hardwired — no TrainingConfig, no config.training."""
    import splade.config.schema as schema
    assert not hasattr(schema, "TrainingConfig"), "TrainingConfig should not exist"

    from splade.config.schema import Config
    config_fields = {f.name for f in Config.__dataclass_fields__.values()}
    assert "training" not in config_fields, "Config should not have a 'training' field"

    from splade.training.loop import train_model
    sig = inspect.signature(train_model)
    params = list(sig.parameters.keys())
    assert "use_df_weighting" not in params, "train_model should not have use_df_weighting param"
    assert "circuit_aware" not in params, "train_model should not have circuit_aware param"
    assert len(params) == 10, f"train_model should have 10 params, got {len(params)}: {params}"
