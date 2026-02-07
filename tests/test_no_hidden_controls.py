import os
import glob
import re
import pytest

CODE_DIR = "splade"
SCRIPTS_DIR = "splade/scripts"

def get_python_files():
    files = glob.glob(os.path.join(CODE_DIR, "**", "*.py"), recursive=True)
    return [f for f in files if "tests" not in f]

@pytest.mark.parametrize("file_path", get_python_files())
def test_no_os_environ_control(file_path):
    """Ensure os.environ is not used for control flow (except specific allowed cases)."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Allow os.environ for CUDA setup in utils/cuda.py or setting env vars, but not GETTING for logic
    if "utils/cuda.py" in file_path:
        return

    # Check for os.environ.get or os.environ[...] usage that looks like a control switch
    # We grep for 'os.environ' and manually review or check for 'get'
    # This is a heuristic.
    
    # We want to forbid os.environ.get("SOME_FLAG")
    matches = re.findall(r'os\.environ\.get\(', content)
    if matches:
        pytest.fail(f"Found os.environ.get usage in {file_path}. Configuration must be via YAML.")

@pytest.mark.parametrize("file_path", glob.glob(os.path.join(SCRIPTS_DIR, "*.py")))
def test_scripts_argparse_clean(file_path):
    """Ensure scripts only accept --config."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find all add_argument calls
    # Regex to capture the argument name (e.g., "--config" or "-c")
    # Matches .add_argument( and then a string starting with -
    matches = re.findall(r'\.add_argument\s*\(\s*(["\']--?[\w-]+["\'])', content)
    
    for arg_name in matches:
        arg_name = arg_name.strip("'\"")
        if arg_name not in ["--config", "--help"]: # --help is implied but usually not explicit, but if explicit it's fine.
             pytest.fail(f"Found extra CLI argument {arg_name} in {file_path}. Only --config is allowed.")