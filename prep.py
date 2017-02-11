import os
from functools import wraps
import pandas as pd
from sklearn.externals import joblib

def _repr_html_(self):
    self = self.copy()

    if self.index.nlevels > 1:
        return None
    else:
        name = self.index.name or 'index'
        if self.columns.name is None:
            self.columns.name = name

        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        show_dimensions = pd.get_option("display.show_dimensions")

        return self.to_html(max_rows=max_rows, max_cols=max_cols,
                            show_dimensions=show_dimensions, notebook=True)

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    pd.DataFrame._repr_html_ = _repr_html_


def cached(name):
    def deco(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs('models', exist_ok=True)
            cache = os.path.join('models', name + '.pkl')
            if os.path.exists(cache):
                return joblib.load(cache)
            result = func(*args, **kwargs)
            joblib.dump(result, cache)
            return result
        return wrapper
    return deco
