import os
import sys

from src.logger import logger
from src.exception import CustomException

import pandas as pd
import numpy as np
import json
from typing import Annotated
from pydantic import BaseModel, Field


class CustomData:

    def get_data_as_dataframe(self):

