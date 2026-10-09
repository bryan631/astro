"""The tablet pages' JavaScript must parse (a syntax error silently breaks the whole UI)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astro import server

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node (CI has it)")
@pytest.mark.parametrize("page", ["index.html", "treeline.html"])
def test_page_script_parses(tmp_path, page):
    scripts = re.findall(r"<script>(.*?)</script>", (WEB / page).read_text(), flags=re.DOTALL)
    assert scripts, "no inline script found"
    js = tmp_path / "page.js"
    js.write_text("\n".join(scripts))
    result = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr


def test_installable_pwa_files_are_served():
    client = TestClient(server.app)
    manifest = client.get("/manifest.json").json()
    assert {i["sizes"] for i in manifest["icons"]} == {"192x192", "512x512"}
    for icon in manifest["icons"]:
        assert client.get("/" + icon["src"]).status_code == 200
    assert "fetch" in client.get("/sw.js").text
