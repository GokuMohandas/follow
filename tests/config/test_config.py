# tests/config/test_config.py
# Test config/config.py components.

from config import config


def test_config():
    assert config.logger.name == "root"
