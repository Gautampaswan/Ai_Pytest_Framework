"""Central settings for the automation suite (paths, timeouts, live endpoints)."""

from __future__ import annotations

from pathlib import Path


class TestSettings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    TESTDATA_DIR: Path = PROJECT_ROOT / "tests" / "testdata"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    DEFAULT_MODEL: str = "llama3.2"
    DEFAULT_LOCATION: str = "Dhaka"
    FAREWELL_MESSAGE: str = "Bye. Have a nice day."

    DEEPEVAL_THRESHOLD: float = 0.5
    DEEPEVAL_STRICT_THRESHOLD: float = 1.0

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TAGS_URL: str = f"{OLLAMA_BASE_URL}/api/tags"
    OLLAMA_HEALTH_TIMEOUT_S: float = 2.0
    LIVE_REQUEST_TIMEOUT_S: float = 30.0

    TEST_TIMEOUT_S: int = 30


settings = TestSettings()
