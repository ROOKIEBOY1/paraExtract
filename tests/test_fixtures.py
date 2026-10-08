import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_preserved_image_matches_recorded_digest():
    image = ROOT / "testdata/source_pocket4_photo_presets.png"
    expected = (ROOT / "testdata/source_pocket4_photo_presets.sha256").read_text().split()[0]
    assert hashlib.sha256(image.read_bytes()).hexdigest() == expected


def test_transcription_keeps_known_ambiguous_lines():
    text = (ROOT / "testdata/manual_transcription.txt").read_text(encoding="utf-8")
    assert "分辨率:1.4K/60fps" in text
    assert "去噪:-0.1" in text
    assert "室内清透感\n曝光:Auto+0.3EV\n50-6400" in text


def test_corpus_contains_all_required_scenarios():
    rows = [json.loads(line) for line in (ROOT / "testdata/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in rows} == {
        "single_night_cyberpunk", "all_scenes_clean", "all_scenes_ocr_noise",
        "similar_concert_scenes", "unlabeled_indoor_iso", "long_repeated_scenes",
    }


def test_fixture_uses_manual_transcription_without_vision_ocr():
    assert not (ROOT / "scripts/ocr_image.swift").exists()
    assert not (ROOT / "testdata/ocr_raw.txt").exists()
    noise = ROOT / "testdata/synthetic_ocr_noise.txt"
    assert noise.is_file()
    rows = [json.loads(line) for line in (ROOT / "testdata/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    noisy = next(row for row in rows if row["id"] == "all_scenes_ocr_noise")
    assert noisy["source_type"] == "synthetic_from_manual_transcription"
    expected_rows = [json.loads(line) for line in (ROOT / "testdata/expected_results.jsonl").read_text(encoding="utf-8").splitlines()]
    noisy_expected = next(row for row in expected_rows if row["id"] == "all_scenes_ocr_noise")
    assert len(noisy_expected["scenes"]) == 4


def test_user_supplied_pocket_posts_are_split_into_six_verbatim_records():
    path = ROOT / "testdata/pocket_user_six_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["id"] for row in rows] == [f"pocket_user_{index:02d}" for index in range(1, 7)]
    assert all(row["source_type"] == "user_supplied_text" for row in rows)
    assert "焦段1.0x～1.3x" in rows[0]["text"]
    assert "胶片影调：CC50%" in rows[1]["text"]
    assert "胶片影调：NC50%" in rows[2]["text"]
    assert "白平衡：4800k 色调+10" in rows[3]["text"]
    assert "ISO：800" in rows[4]["text"]
    assert "ISO：100-800" in rows[5]["text"]
