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


def test_installable_pwa_files_are_served():

    from fastapi.testclient import TestClient

    from astro import server

    client = TestClient(server.app)
    manifest = client.get("/manifest.json").json()
    assert {i["sizes"] for i in manifest["icons"]} == {"192x192", "512x512"}
    for icon in manifest["icons"]:
        assert client.get("/" + icon["src"]).status_code == 200
    assert "fetch" in client.get("/sw.js").text
