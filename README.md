# Project Workflow

## 1. Training
## 2. Evaluating
## 3. Predicting

All three processes are implemented in a single Python script svsToExpr.py:  
**`
from svsToExpr import process_svs_to_expression
result=process_svs_to_expression("./","test.svs","BRCA")
`**

We provide a demonstration script showing how to use the workflow:  
**`demo.py`**

---

## 4. Website Implementation

This repository contains the source code for:  
**https://www.hbpding.com/ImageDependency/**

### Deployment Recommendations
- **Server**: Apache2 on Debian
- **Environment**: Python3 required
