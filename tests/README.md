# Weather AI Agent — Pytest Automation Framework

Industry-standard Python test suite for the Weather AI Agent CLI.

## Layout

```text
tests/
├── conftest.py              # shared fixtures, markers, session bootstrap
├── config/settings.py       # paths, timeouts, live endpoints
├── testdata/                # JSON/YAML fixtures (data-driven cases)
├── support/                 # factories, fakes, CLI harness, assertions
├── unit/                    # isolated module tests (default, fast)
├── integration/             # cross-module tests; live tests opt-in
└── e2e/                     # full CLI conversation flows
```

## Layers

| Marker | What it covers | Network |
| --- | --- | --- |
| `unit` | `lib` helpers, mapping, CLI branches with fakes | No |
| `integration` | Weather → analyze → format pipeline | No (mocked) |
| `e2e` | `main.main()` conversations via `WeatherAgentHarness` | No |
| `live` | Real Ollama and weather provider | Yes, opt-in |

Default run excludes `live`.

## Commands

```bash
pip install -r requirements-test.txt

pytest                          # unit + integration + e2e, coverage + HTML report
pytest -m unit                  # fast isolated tests
pytest -m e2e                   # CLI conversation flows
pytest -m live                  # requires Ollama on localhost:11434
pytest tests/unit/test_ai.py -k isBye
```

Reports:

- Terminal coverage: enabled by default
- HTML: `reports/report.html`
- JUnit (CI): `reports/junit.xml`

## Design rules

- **AAA** (Arrange / Act / Assert) in every test
- **No live I/O** unless marked `live` and skipped when the service is down
- **One weather snapshot per conversation** — matches production (`getweather` once, then chat)
- **Fakes over mocks** for OpenAI-compatible completions and `python_weather.Client`
- **Harness** drives stdin/stdout like a UI driver: `WeatherAgentHarness.run([...])`
- **Product contracts** live in `tests/support/assertions.py` and `WEATHER_SCHEMA_KEYS`

## Environment

Copy `.env.sample` to `.env`. Tests set `MODEL=llama3.2` when it is unset. `.env` is gitignored and must never be committed.
