from text_extractor.evaluation.monitoring import process_tree_rss, summarize_latencies


def test_latency_percentiles_use_nearest_rank():
    summary = summarize_latencies([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary.mean_ms == 5.5
    assert summary.p50_ms == 5
    assert summary.p95_ms == 10


def test_rss_monitor_falls_back_to_current_process_when_child_listing_is_blocked():
    class Memory:
        rss = 1234

    class Process:
        def children(self, recursive):
            raise PermissionError("sysctl blocked")

        def memory_info(self):
            return Memory()

        def is_running(self):
            return True

    assert process_tree_rss(Process()) == 1234
