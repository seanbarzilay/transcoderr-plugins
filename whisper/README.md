# whisper

Generates a sidecar `.srt` for the post-transcode output file using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). GPU is
auto-detected; falls back to CPU.

## Step

`whisper.transcribe` — runs after `output` in a flow.

## Config

| Key | Default | Description |
|---|---|---|
| `model` | `large-v3-turbo` | faster-whisper model name |
| `language` | `auto` | ISO 639-1 or `auto` |
| `skip_if_exists` | `true` | Bail if a sidecar already exists |
| `compute_type` | `auto` | CTranslate2 compute type; auto picks `float16` on GPU, `int8` on CPU |

## Output

Writes `<basename>.<lang>.srt` next to the input file. Sets
`steps.whisper` in ctx:

```json
{
  "subtitle_path": "/data/movies/Movie.en.srt",
  "language": "en",
  "model": "large-v3-turbo",
  "duration_sec": 42.7
}
```

(Filled out fully in Task 10.)
