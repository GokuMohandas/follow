# tests/model/behavioral.py
# Behavioral testing components.

from pathlib import Path

import pytest

from config import config
from tagifai import main, predict


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@pytest.mark.parametrize(
    "texts, tags",
    [
        (
            ["Deep learning accelerated NLP.", "Deep learning catalyzed NLP."],
            ["natural-language-processing"],
        ),  # verb injection
        (
            ["Generative adverseril networks", "Generative adversarial networks"],
            ["natural-language-processing"],
        ),  # misspelling
    ],
)
def test_invariance(texts, tags, artifacts):
    """INVariance via verb injection, misspelling, etc. (changes should not affect outputs)."""
    # INVariance (changes should not affect outputs)
    results = predict.predict(texts=texts, artifacts=artifacts)
    assert [set(tags).issubset(set(result["predicted_tags"])) for result in results]


@pytest.mark.parametrize(
    "texts, tags",
    [
        (
            [
                "A TensorFlow implementation of transformers.",
                "A PyTorch implementation of transformers.",
            ],
            ["tensorflow", "pytorch"],
        )
    ],
)
def test_directional(texts, tags, artifacts):
    """DIRectional expectations (changes with known outputs)."""
    results = predict.predict(texts=texts, artifacts=artifacts)
    assert [tags[i] in result["predicted_tags"] for i, result in enumerate(results)]


@pytest.mark.parametrize(
    "text, tags",
    [
        ("Transformers have revolutionized machine learning.", ["transformers"]),
        ("GNNs have revolutionized machine learning.", ["graph-neural-networks"]),
    ],
)
def test_mft(text, tags, artifacts):
    """# Minimum Functionality Tests (simple input/output pairs)."""
    results = predict.predict(texts=[text], artifacts=artifacts)
    assert [set(result["predicted_tags"]).issubset(set(tags)) for result in results]
