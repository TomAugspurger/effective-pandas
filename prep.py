import pandas as pd

def _repr_html_(self):
    if self.index.nlevels > 1:
        return None
    else:
        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        show_dimensions = pd.get_option("display.show_dimensions")

        return self.to_html(max_rows=max_rows, max_cols=max_cols,
                            show_dimensions=show_dimensions, notebook=True)

pd.DataFrame._repr_html_ = _repr_html_
