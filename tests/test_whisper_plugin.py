"""Tests for whisper/plugin.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the whisper plugin's plugin.py importable.
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "whisper"
sys.path.insert(0, str(PLUGIN_DIR))
import plugin  # noqa: E402


class ImportSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(plugin, "DEFAULT_CONFIG"))
        self.assertEqual(plugin.DEFAULT_CONFIG["model"], "large-v3-turbo")
        self.assertEqual(plugin.DEFAULT_CONFIG["language"], "auto")
        self.assertTrue(plugin.DEFAULT_CONFIG["skip_if_exists"])
        self.assertEqual(plugin.DEFAULT_CONFIG["compute_type"], "auto")


class FmtTsTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(plugin.fmt_ts(0.0), "00:00:00,000")

    def test_subsecond(self):
        self.assertEqual(plugin.fmt_ts(1.234), "00:00:01,234")

    def test_minutes_and_seconds(self):
        self.assertEqual(plugin.fmt_ts(61.5), "00:01:01,500")

    def test_hours(self):
        self.assertEqual(plugin.fmt_ts(3661.5), "01:01:01,500")

    def test_rounds_to_nearest_millisecond(self):
        # 0.0005 sits exactly on the boundary; banker's rounding is fine
        # — tests for the ms = round(secs * 1000) behaviour.
        self.assertEqual(plugin.fmt_ts(0.001), "00:00:00,001")
        self.assertEqual(plugin.fmt_ts(0.0014), "00:00:00,001")
        self.assertEqual(plugin.fmt_ts(0.0016), "00:00:00,002")


from collections import namedtuple

_Segment = namedtuple("_Segment", ["start", "end", "text"])


class FormatSrtTests(unittest.TestCase):
    def test_empty_iterable_returns_empty_string(self):
        self.assertEqual(plugin.format_srt([]), "")

    def test_single_segment(self):
        out = plugin.format_srt([_Segment(1.0, 2.0, "Hello world.")])
        self.assertEqual(
            out,
            "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n\n",
        )

    def test_two_segments_numbered_and_separated(self):
        segs = [
            _Segment(1.0, 2.0, "One."),
            _Segment(2.5, 3.5, "Two."),
        ]
        self.assertEqual(
            plugin.format_srt(segs),
            "1\n00:00:01,000 --> 00:00:02,000\nOne.\n\n"
            "2\n00:00:02,500 --> 00:00:03,500\nTwo.\n\n",
        )

    def test_text_is_stripped(self):
        out = plugin.format_srt([_Segment(0.0, 1.0, "  spaced  ")])
        self.assertIn("spaced\n\n", out)
        self.assertNotIn("  spaced  \n", out)


class WriteSrtAtomicallyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_exact_bytes(self):
        target = self.dir / "Movie.en.srt"
        plugin.write_srt_atomically(target, "1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n")
        self.assertEqual(
            target.read_text(),
            "1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n",
        )

    def test_no_tmp_left_behind_on_success(self):
        target = self.dir / "Movie.en.srt"
        plugin.write_srt_atomically(target, "x")
        self.assertFalse((self.dir / "Movie.en.srt.tmp").exists())

    def test_overwrites_existing_target(self):
        target = self.dir / "Movie.en.srt"
        target.write_text("OLD")
        plugin.write_srt_atomically(target, "NEW")
        self.assertEqual(target.read_text(), "NEW")

    def test_creates_parent_dir_if_missing(self):
        target = self.dir / "nested" / "Movie.en.srt"
        with self.assertRaises(FileNotFoundError):
            # We do NOT auto-create parents — the caller (transcribe) is
            # responsible. Confirm the contract.
            plugin.write_srt_atomically(target, "x")


class FindExistingSidecarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "Movie.mkv"
        self.video.write_text("")  # empty placeholder

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_none_when_no_sidecar_exists(self):
        self.assertIsNone(plugin.find_existing_sidecar(self.video, "en"))
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_exact_match_when_language_fixed(self):
        sidecar = self.dir / "Movie.en.srt"
        sidecar.write_text("")
        self.assertEqual(plugin.find_existing_sidecar(self.video, "en"), sidecar)

    def test_no_match_when_language_fixed_but_only_other_lang_present(self):
        (self.dir / "Movie.fr.srt").write_text("")
        self.assertIsNone(plugin.find_existing_sidecar(self.video, "en"))

    def test_any_lang_match_when_language_is_none(self):
        sidecar = self.dir / "Movie.fr.srt"
        sidecar.write_text("")
        # find_existing_sidecar with None returns the FIRST match found.
        # The exact path (which one of several) is unimportant — any
        # truthy return value is treated as "skip".
        result = plugin.find_existing_sidecar(self.video, None)
        self.assertEqual(result, sidecar)

    def test_no_match_for_different_basename(self):
        (self.dir / "OtherMovie.en.srt").write_text("")
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_does_not_match_the_video_file_itself(self):
        # Movie.mkv exists but isn't a .srt — must not be confused for one.
        self.assertIsNone(plugin.find_existing_sidecar(self.video, None))

    def test_preserves_dot_separated_quality_tags_in_filename(self):
        # Files like "Movie.2024.1080p.mkv" must look for sidecars at
        # "Movie.2024.1080p.<lang>.srt", not "Movie.2024.<lang>.srt".
        complex_video = self.dir / "Movie.2024.1080p.mkv"
        complex_video.write_text("")
        sidecar = self.dir / "Movie.2024.1080p.en.srt"
        sidecar.write_text("")
        # Wrong-prefix sidecar that must NOT be matched.
        (self.dir / "Movie.2024.en.srt").write_text("")
        self.assertEqual(
            plugin.find_existing_sidecar(complex_video, "en"),
            sidecar,
        )


if __name__ == "__main__":
    unittest.main()
