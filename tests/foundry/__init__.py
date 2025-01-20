"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import sys

# Expose logging messages.
logger = logging.getLogger()
logger.setLevel("INFO")
stream_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
