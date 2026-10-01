import pytest

from src.serving.decision import churn_class_from_probability


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.3499, 0), (0.35, 1), (0.3501, 1)],
)
def test_churn_class_matches_inclusive_threshold(probability: float, expected: int):
    assert churn_class_from_probability(probability, 0.35) == expected


@pytest.mark.parametrize(
    "probability,threshold",
    [(-0.01, 0.35), (1.01, 0.35), (float("nan"), 0.35), (0.5, -0.1), (0.5, 1.1), (0.5, float("nan"))],
)
def test_churn_class_rejects_values_outside_probability_range(probability: float, threshold: float):
    with pytest.raises(ValueError):
        churn_class_from_probability(probability, threshold)