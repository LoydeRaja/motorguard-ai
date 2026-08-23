from src.features import build_preprocessor


def test_preprocessor_has_numeric_and_categorical_blocks():
    preprocessor = build_preprocessor()

    transformer_names = {
        name for name, _, _ in preprocessor.transformers
    }

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names