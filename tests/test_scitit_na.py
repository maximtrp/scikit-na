from src import scikit_na as na
from pandas import read_csv
import pytest


@pytest.fixture
def data():
    return read_csv("./tests/titanic_dataset.csv")


def test_summary(data):
    summary = na.summary(data)

    na_counts_correct = all(data.isna().sum() == summary.loc["na_count"])
    na_pct_correct = all(
        (data.isna().sum() / data.shape[0] * 100).round(2)
        == summary.loc["na_pct_per_col"]
    )
    na_pct_total = (data.isna().sum() / data.isna().sum().sum() * 100).round(2)
    na_pct_total_correct = all(na_pct_total == summary.loc["na_pct_total"])
    rows_after_dropna_correct = all(
        (data.shape[0] - data.isna().sum()) == summary.loc["rows_after_dropna"]
    )
    rows_after_dropna_pct_correct = all(
        ((data.shape[0] - data.isna().sum()) * 100 / data.shape[0]).round(2)
        == summary.loc["rows_after_dropna_pct"]
    )

    na_unique1 = (
        ((data.isna().sum(axis=1) == 1) & data.loc[:, "Age"].isna()).sum().item()
    )
    na_unique_correct1 = na_unique1 == 19
    na_unique_pct_correct1 = (na_unique1 / data.isna().sum()["Age"] * 100).round(
        2
    ) == summary.loc["na_unique_pct_per_col", "Age"]
    na_unique_correct2 = (
        (data.isna().sum(axis=1) == 1) & data.loc[:, "Cabin"].isna()
    ).sum().item() == 529
    na_unique_correct3 = (
        (data.isna().sum(axis=1) == 1) & data.loc[:, "Survived"].isna()
    ).sum().item() == 0
    assert na_counts_correct
    assert na_pct_correct
    assert na_pct_total_correct
    assert rows_after_dropna_correct
    assert rows_after_dropna_pct_correct
    assert na_unique_correct1
    assert na_unique_pct_correct1
    assert na_unique_correct2
    assert na_unique_correct3
