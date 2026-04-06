# Pipeline API smoke test

`test_pipeline_api.py` exercises the FastAPI app without starting RTSP workers.

**Full operations guide (pytest, cURL, SSH, cashier cases):**  
[docs/VISION_PIPELINE_README.md](../../docs/VISION_PIPELINE_README.md)

## Run

From the repository root:

```bash
python3 -m pytest tests/pipeline_test/test_pipeline_api.py -v
python3 -m pytest tests/pipeline_test/ -v
```

Fixtures: `conftest.py` (includes an `ultralytics` stub).
