# AI-Powered-Radiology-Assistant

```text
AI-Powered-Radiology-Assistant/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── download_weights.py        # (optional) fetch model weights automatically
├── src/                       # all application code in one place
│   ├── __init__.py
│   ├── app.py                 # Gradio interface & main pipeline
│   ├── utils.py               # helper functions (model loading, downloads)
│   ├── classification/        # tumor classification module
│   │   ├── __init__.py
│   │   └── transunet.py       # TransUNetClassifier definition
│   ├── segmentation/          # tumor segmentation module
│   │   ├── __init__.py
│   │   └── unet.py            # build_unet, metrics (dice, iou)
│   └── report/                # report-generation logic
│       ├── __init__.py
│       └── generator.py       # extract_structured_info, LLM wrapper
└── notebooks/                 # Jupyter notebooks for training & experiments
    ├── Tumor_Classification_Transunet_code.ipynb
    ├── Tumor_Segmentation_code.ipynb
    ├── Generate_Reports.ipynb
    ├── LLM_Train_Model.ipynb
    └── Model_Performance_Comparison.ipynb
```
