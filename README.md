This project performs data processing and utilization of XGBoost classifiers. It is not expected to yield financial value. An improved version may yield modest value as a time-saver when sifting through companies to evaluate, provided it delivers reliable results over several subsequent years.

**Setup & Usage**
```bash
pip install pandas numpy xgboost scikit-learn requests
```
- Create project folders: `data/` and `output/`.
- Utilize `retrieve_data.py` to (1) retrieve data via FMP API; or (2) load existing data, given that the required content exists in `data/`.

Run the following sequentially from the project root:
  ```bash
  python retrieve_data.py
  ```
  - Produces `output/yearly_dataset.csv`
  ```bash
  python process_data.py
  ```
  - Produces `output/processed_dataset.csv`, `output/processed_2024_dataset.csv`, `output/spy_data.csv`
  ```bash
  python xgb_1.py
  ```
  - Produces `output/xgb_1_predictions_2024.csv`
  ```bash
  python xgb_2.py
  ```
  - Produces `output/xgb_2_predictions_2024.csv`

**Notebooks**
- Use `process_data_notebook` to inspect data processing.
- Use `xgb_1_notebook` to view the model's predictions.
- Use `xgb_2_notebook` to inspect precision/recall trade-off.

**Results**
- `xgb_1.py` emphasizes precision (0.90 train, 0.75 test) while heavily sacrificing recall (<0.10) when flagging outperformers.
- `xgb_2.py` emphasizes F1 (0.817 train, 0.798 test) when flagging underperformers. Prioritizing recall instead may be a more sensible and conservative approach.
- `xgb_2.py` is intended to complement `xgb_1.py` by flagging potential underperformers among predicted outperformers.
- As of September 6, 2025, only 3 stocks (CEG, SMCI, NVDA) have been assigned a >60% chance of beating the index; they have outperformed the index by an average of 15.8%.
- `xgb_2.py` has not proved reliable, as stocks assigned a >95% chance of underperforming have, on average, matched the index's positive return.

**Project Structure**
```
xgb_sp500/
├── data/                       
├── output/                     
├── retrieve_data.py            
├── process_data.py
├── process_data_notebook.ipynb           
├── xgb_1.py
├── xgb_1_notebook.ipynb         
├── xgb_2.py
├── xgb_2_notebook.ipynb               
└── README.md                
```
