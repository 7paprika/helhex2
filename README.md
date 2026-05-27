# Helical Tube Heat Exchanger Designer

Streamlit engineering calculator for preliminary sizing and reporting of a helical coil heat exchanger.

## What was hardened

- Extracted pure calculation/report helpers into `calculator.py`
- Added validation for nonphysical and inconsistent inputs
- Added automated pytest coverage for core scenarios and regressions
- Fixed tube material/report state mismatch
- Disabled report export when the scenario is invalid
- Added project metadata, CI, and sensible git ignores

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python3 -m pytest
```

## Notes

This tool is intended for preliminary engineering checks, not final code-stamped design. Validate final thermal, hydraulic, mechanical, and fabrication assumptions before procurement or construction.
