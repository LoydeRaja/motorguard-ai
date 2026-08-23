from src.motorguard_ai.features import build_preprocessor


def test_build_preprocessor_creates_transformers():
    preprocessor = build_preprocessor()
    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "num" in transformer_names
    assert "cat" in transformer_names
