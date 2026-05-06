# whisperx

Forced phoneme alignment for whisper-generated subtitles via [WhisperX](https://github.com/m-bain/whisperX).
Provides two steps:

- `whisperx.align` — drop-in after `whisper.transcribe`. Re-times an existing `.srt` using wav2vec2 forced alignment.
- `whisperx.transcribe_aligned` — full pipeline (transcribe + align) standalone.

(Operator-facing usage docs land in Task 7.)
