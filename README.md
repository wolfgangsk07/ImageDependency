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
Linux:
```
sudo apt-get install openslide-tools
pip install -r requirements.txt
```
