# In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code


---

## Overview

This paper proposes an in-context learning (ICL) approach to estimate the functional correctness of LLM-generated code in the absence of test cases. It ranks generated code solutions based on their quality based on functional correctness. 

---

### Key Features

-  Similarity-based few-shot example selection
-  Functional correctness of LLM-Generated code 
-  nDCG@K based evaluation for ranking quality
-  Supports cross-lingual datasets 

---

### Environment Setup
To run the experiments create a conda environment and install the required libraries.
```bash
pip install -r requirement.txt
```


### Baselines
To run the unsupervised experiments make the following changes
- In the `config.properties` change `task` and `input_file` as needed
  - `els` for ELS (Embedding Level Similarity)
  - `tls` for TLS (Token Level Similarity)
  - `llm` for Zero-shot LLM inferencing
  - check the `data` folder for the input file
 
```bash
python unsup-prediction.py config.properties
```

### Fewshot Method (Proposed Method) 

#### Find best nDCG on validatiuon set 
To run the fewshot code quality estimation code find the best `k` value on the validation set. <br>
`MBPP` data is split into train and validation set in the ratio 90:10. To find the best nDCG value 
```bash
python train.py --config-train.json
```

#### Run on the test set 
To get the results on the test set `HumanEval` or `MBJP` 
```bash
python test.py --config-test.json
```



