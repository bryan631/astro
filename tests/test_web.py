"""The tablet page's JavaScript must parse (a syntax error silently breaks the whole UI)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node (CI has it)")
def test_page_script_parses(tmp_path):
    scripts = re.findall(r"<script>(.*?)</script>", PAGE.read_text(), flags=re.DOTALL)
    assert scripts, "no inline script found"
    js = tmp_path / "page.js"
    js.write_text("\n".join(scripts))
    result = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr
