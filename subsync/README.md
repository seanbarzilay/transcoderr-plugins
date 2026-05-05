# subsync

`subsync.align` — re-times an out-of-sync `.srt` against the audio of a
video file using [ffsubsync](https://github.com/smacke/ffsubsync).
Designed to drop in after `whisper.transcribe` to clean up timing drift
in the whisper-generated sidecar.

(Operator-facing usage docs land in Task 6.)
