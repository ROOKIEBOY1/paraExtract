from text_extractor.evaluation.benchmarking import build_length_text, run_latency_matrix


def test_build_length_text_repeats_source_to_requested_character_count():
    assert build_length_text("abc", 8) == "abc\nabc\n"


def test_latency_matrix_reuses_loaded_backend_and_uses_requested_batches():
    calls = []

    class Backend:
        def extract(self, texts):
            calls.append(list(texts))
            return [{} for _ in texts]

    ticks = iter(index / 1000 for index in range(100))
    rows = run_latency_matrix(
        Backend(),
        "abcdef",
        lengths=[3, 6],
        batch_sizes=[1, 2],
        warmups=1,
        runs=2,
        clock=lambda: next(ticks),
    )

    assert len(rows) == 4
    assert [len(call) for call in calls] == [1, 1, 1, 2, 2, 2] * 2
    assert all(len(row["latencies_ms"]) == 2 for row in rows)
    assert rows[0]["length_chars"] == 3
    assert rows[-1]["batch_size"] == 2
