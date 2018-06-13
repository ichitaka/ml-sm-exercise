import numpy as np
import pandas as pd
import scipy as sci
from scipy import stats

df = pd.DataFrame([4,4,2,4,1,1])

print(stats.chisquare(df))