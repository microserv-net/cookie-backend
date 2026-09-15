from cookie_backend.parsing import find_json_object, spoken_text, strip_reasoning


def test_plain_object():
    assert find_json_object('{"kind": "chat"}') == {"kind": "chat"}


def test_fenced_object_with_preamble():
    text = 'Sure! Here you go:\n```json\n{"kind": "task", "weight": "heavy"}\n```\nHope that helps.'
    assert find_json_object(text) == {"kind": "task", "weight": "heavy"}


def test_reasoning_blocks_never_parse_as_the_answer():
    text = '<think>The user wants {"kind": "chat"} probably</think>{"kind": "task"}'
    assert find_json_object(text) == {"kind": "task"}


def test_unterminated_reasoning_is_discarded():
    assert strip_reasoning("answer <think>still going") == "answer"


def test_nested_objects_and_braces_in_strings():
    text = '{"steps": [{"what": "say {hello}", "done_when": "it is said"}], "say": "ok"}'
    parsed = find_json_object(text)
    assert parsed["steps"][0]["what"] == "say {hello}"


def test_nothing_usable_returns_none():
    assert find_json_object("I'm not sure what you mean.") is None
    assert find_json_object("") is None


def test_spoken_text_removes_what_a_synthesiser_would_read_aloud():
    text = "<think>hmm</think>**Done.** Here is the code:\n```py\nx=1\n```\n- it works"
    spoken = spoken_text(text)
    assert "think" not in spoken and "*" not in spoken and "x=1" not in spoken
    assert spoken.startswith("Done.")
