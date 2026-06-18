from arion_agent.environments.file import ops
from arion_agent.summarization.compress import DEFAULT_SUMMARY_BUDGET
from arion_agent.summarization.sections import format_configured_skills
from pathlib import Path
import tempfile

assert DEFAULT_SUMMARY_BUDGET == 3600
with tempfile.TemporaryDirectory() as td:
    ws = Path(td)
    err = ops.list_files("../../etc/passwd", ws)
    assert "PathConfinement" in err
    assert "Workspace root (absolute):" in err
text = format_configured_skills()
assert "None configured" in text
print("OK", DEFAULT_SUMMARY_BUDGET)
