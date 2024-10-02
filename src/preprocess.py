class MissingHandler(object):
    def handle_missing_values(self, data_frame):
        """Fill missing values."""
        return data_frame.ffill().bfill()
