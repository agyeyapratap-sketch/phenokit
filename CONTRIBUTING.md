# Contributing to phenokit

Thanks for your interest! Bug reports, feature requests, datasets and pull requests are all welcome.

## Reporting problems
Open a GitHub issue with:
- what you ran (command or code snippet) and the phenokit version (`python -c "import phenokit; print(phenokit.__version__)"`);
- what you expected and what happened (full traceback if there is one);
- your OS and Python version, and the sensor/data format if relevant.

## Development setup
```bash
git clone https://github.com/agyeyapratap-sketch/phenokit.git
cd phenokit
pip install -e ".[dev]"      # add ",deep" for the PyTorch models
python -m pytest
```

## Pull requests
- Keep changes focused; one topic per pull request.
- Add or update tests: every new QC check or estimator should have a test with a known answer
  (an analytic value, a round-trip, or an injected fault it must detect).
- Keep core dependencies light (NumPy, SciPy, pandas, scikit-learn, Matplotlib, Pillow, PyYAML).
  Heavier libraries belong behind optional extras.
- Update `CHANGELOG.md` under an "Unreleased" heading.
- Describe in the pull request whether generative AI tools were used and how you checked their output.

## Support
phenokit is maintained on a best-effort basis by its author. Issues are usually answered within a couple of weeks.
