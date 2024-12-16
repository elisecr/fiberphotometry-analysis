import warnings

import panel as pn


def _warning(message, category=UserWarning, filename="", lineno=-1, line=None):
    return "WARNING: " + str(message) + "\n"


warnings.formatwarning = _warning
pn.extension()
