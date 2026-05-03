# Examples

Minimal scripts that load a released SRT adapter from Hugging Face and run
inference. They require only the `srt-adapter` install plus an internet
connection (no GPU strictly required, but recommended for the 7B backbone).

```bash
pip install -e ..
python load_and_score.py --repo RiverRider/srt-adapter-v1.0 --text "meaning forks here"
python encode_sentences.py --repo RiverRider/srt-adapter-v1.0 \
    --sentences "the bank by the river" "the bank gave me a loan"
```
