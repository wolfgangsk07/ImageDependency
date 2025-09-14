# Project Workflow

## 1. Training
## 2. Evaluating
## 3. Predicting

All predicting processes are implemented in a single Python script svsToExpr.py:  
```
from svsToExpr import process_svs_to_expression
result=process_svs_to_expression("./","test.svs","BRCA")
```

---

## 4. Website Implementation

This repository contains the source code for:  
**https://www.hbpding.com/ImageDependency/**

### Deployment Recommendations
- **Server**: Apache2 on Debian
- **Environment**: Python3 required:
```
import sqlite3
import traceback
from datetime import datetime
from svsToExpr import process_svs_to_expression
import pandas as pd
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import logging
import h5py
from tqdm import tqdm
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
import string
import time
import timm
from torchvision import transforms
import torchvision.models as models
import numpy as np
import torch
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.cluster import KMeans
from src.agentAttention import AgentAttention
```
