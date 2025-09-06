# 🚚 Hyperlocal Delivery Time Predictor

**Short summary**  
A beginner-friendly end-to-end ML project that predicts hyperlocal delivery time (minutes) using features such as distance, traffic, weather, pickup hour, and driver experience. Built with Python, scikit-learn and Streamlit. Includes dataset generation, training pipeline, EDA with visualizations, and a web app deployed on Streamlit Cloud.

---

## Demo
- Live app (Streamlit): **`https://hyperlocal-delivery-time-fpdcyq262wbhbipcxu9yxc.streamlit.app/`**  
  _Replace with your actual Streamlit app URL after deployment._


---

## What this project contains

HyperlocalDeliveryTime/
├─ data/ # synthetic dataset (CSV)
│ └─ delivery_data.csv
├─ models/ # trained pipeline is saved here on first-run (ignored in repo)
├─ reports/
│ ├─ figures/ # EDA images (PNGs)
│ └─ EDA_report.md # short EDA summary
├─ src/
│ ├─ data_generation.py # creates synthetic dataset (6000 rows)
│ ├─ train_model.py # trains pipeline and saves pipeline
│ └─ eda.py # generates EDA figures + report
├─ app.py # Streamlit app
├─ requirements.txt # Python dependencies
├─ .gitignore
└─ README.md



---

## Why this is useful
- Shows knowledge of the full ML lifecycle: data generation/cleaning → model training → saving → web UI → deployment → EDA.  
- Uses practical features relevant to courier/delivery use cases (traffic, weather, stops, distance).  
- Recruiter-friendly: easy to run locally and to demo online.

---

## Quick start (Windows / PowerShell / VS Code) — copy & paste

**Open VS Code terminal (PowerShell), then run:**

```powershell
# 1) Activate your venv
cd path\to\HyperlocalDeliveryTime
.\venv\Scripts\Activate.ps1

# 2) Install (if you added or updated requirements)
pip install -r requirements.txt

# 3) (Re)generate dataset (if needed)
python src/data_generation.py

# 4) Run EDA to create figures and report
python src/eda.py

# 5) Train model locally (creates models/delivery_time_pipeline.joblib)
python src/train_model.py

# 6) Run the Streamlit app locally
streamlit run app.py

## Notes
This project is beginner-friendly and good for interviews. For production, collect real delivery logs and add monitoring.

## License
This project is released under the MIT License — feel free to reuse and adapt.

## How to cite / contact

Author: Arshad Murtaza
GitHub: https://github.com/arshadmurtaza03/hyperlocal-delivery-time.git
Email: arshadmurtaza2016@gmail.com

