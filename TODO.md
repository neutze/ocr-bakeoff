# TODO

- [x] First full run of every lane on `samples` (2026-09-25): all 16 lanes ran; timings in README. Three engines tie at the top on the samples (qwen8b, qwen4b, paddle).
- [x] Real runs (2026-09-28/29): 20 + 40 real born-digital documents and 15 + 30 real scans from the owner's Downloads, picked with `pick`; aggregates in the README, "What it found on one archive".
- [ ] Try `--reference gdrive` end to end with a real OAuth client; the self-test covers it only against a fake Drive. (Not available to the owner; `--reference consensus` is used instead.)
- [x] Each model's licence checked (2026-10-26): listed in `spec/data/Engine Lanes.md` and the README.
- [x] `prompts/glm_custom.txt` ships publicly (owner, 2026-10-26).
- [x] Public, as release commits only; the history stays in the private repository (owner, 2026-10-26).
- [ ] Consider a `--fallback` lane option: re-read a page with a classic engine when a model's output loops or hits the cap (Qwen3-VL looped on 1–3 of 30 real scans).
- [ ] List the worker engines' packages and models (`doctr`, `easyocr`) in the first-run install list; today `uv` and the libraries fetch them on first use without the prompt.
- [ ] Time a worker engine's start-up apart from its first page.
- [ ] Add speed figures for the lanes added on 2026-10-04 to the README once the October 2026 evaluation has finished.
- [ ] GLM-OCR with its own layout step (regions from PP-DocLayout-V3), which is how the model is meant to be used; the existing GLM lanes give it whole pages, bands or Apple Vision blocks.
