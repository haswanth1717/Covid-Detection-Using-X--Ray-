import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/MINOR/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
