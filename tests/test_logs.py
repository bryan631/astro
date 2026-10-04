import json
import logging

from astro import logs


def test_json_lines_with_data_and_pruning(tmp_path):
    for _ in range(4):
        path = logs.setup(tmp_path, keep=3)
    logging.getLogger("astro.test").info("said", extra={"data": {"text": "push left"}})
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("astro.test").exception("tick failed")
    for h in logging.getLogger("astro").handlers:
        h.flush()
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["event"] == "said" and lines[0]["text"] == "push left"
    assert lines[1]["level"] == "ERROR" and "ValueError: boom" in lines[1]["exc"]
    assert len(list((tmp_path / logs.LOG_DIR).glob("astro-*.jsonl"))) == 3
    logging.getLogger("astro").handlers.clear()
