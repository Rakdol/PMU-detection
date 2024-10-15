import pandas as pd


class MissingHandler(object):
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        return df.ffill().bfill()


if __name__ == "__main__":

    df = pd.DataFrame({"a": [0, None, 1, 1, None]})
    assert df.isnull().sum().values[0] != 0

    handler = MissingHandler()
    assert handler.handle_missing_values(df).isnull().sum().values[0] == 0
