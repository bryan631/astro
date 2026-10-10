import json
import logging
import threading

from astro import logs


def test_json_lines_with_data_and_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)  # setup replaces it
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


def test_a_crashed_thread_lands_in_the_log(tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)  # undone after the test
    path = logs.setup(tmp_path)

    def crash():
        raise KeyError("bug")

    t = threading.Thread(target=crash, name="auto-solve")
    t.start()
    t.join()
    for h in logging.getLogger("astro").handlers:
        h.flush()
    entry = json.loads(path.read_text().splitlines()[-1])
    assert entry["event"] == "thread crashed" and entry["thread"] == "auto-solve"
    assert "KeyError: 'bug'" in entry["exc"]
    logging.getLogger("astro").handlers.clear()
