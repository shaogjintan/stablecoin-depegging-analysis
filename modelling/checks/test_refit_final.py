import pandas as pd
import pytest

from modelling.refit_final import check_refit_partition


def _rows(start, end, coin="USDC"):
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.DataFrame(dict(coin=coin, hour=hours, y_true=0))


def _parts():
    inner_train = _rows("2022-06-01", "2022-12-30 20:00")
    inner_val = _rows("2023-01-02", "2023-12-30 20:00")
    gap = _rows("2022-12-30 21:00", "2023-01-01 23:00")
    test = _rows("2024-01-02", "2024-02-01")
    train = pd.concat([inner_train, gap, inner_val], ignore_index=True)
    return train, inner_train, inner_val, test


def test_valid_final_partition_reports_added_rows():
    train, inner_train, inner_val, test = _parts()
    stats = check_refit_partition(train, inner_train, inner_val, test, 2024)
    assert stats["validation_rows_added"] == len(inner_val)
    assert stats["gap_rows_added"] == len(train) - len(inner_train) - len(inner_val)


def test_rejects_test_rows_in_training():
    train, inner_train, inner_val, test = _parts()
    with pytest.raises(ValueError):
        check_refit_partition(pd.concat([train, test.head(1)]), inner_train, inner_val, test, 2024)


def test_rejects_rows_inside_purge_window():
    train, inner_train, inner_val, test = _parts()
    late = _rows("2023-12-31 00:00", "2023-12-31 00:00")
    with pytest.raises(ValueError, match="purge"):
        check_refit_partition(pd.concat([train, late]), inner_train, inner_val, test, 2024)


def test_rejects_final_set_without_validation_year():
    train, inner_train, inner_val, test = _parts()
    with pytest.raises(ValueError, match="practice train and validation"):
        check_refit_partition(inner_train, inner_train, inner_val, test, 2024)


def test_rejects_test_inside_embargo():
    train, inner_train, inner_val, test = _parts()
    early = pd.concat([_rows("2024-01-01", "2024-01-01"), test])
    with pytest.raises(ValueError, match="embargo"):
        check_refit_partition(train, inner_train, inner_val, early, 2024)
