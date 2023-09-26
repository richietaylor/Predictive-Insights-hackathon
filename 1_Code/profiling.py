from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ydata_profiling import ProfileReport



df = pd.read_csv("processed_train.csv")

profile = ProfileReport(df, title="Profiling Report")

profile.to_file("report.html")