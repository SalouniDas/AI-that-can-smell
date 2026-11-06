<div align="center">
  <h1>👃 AI That Can Smell</h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=FF4B4B&center=true&vCenter=true&width=800&lines=Building+an+AI+Nose;Simulated+Gas+Sensors+%26+Molecular+GNNs;Powered+by+OpenPOM%2C+Streamlit+%26+Scikit-Learn" alt="Typing SVG" />

  **A molecular machine learning pipeline that simulates an electronic nose to predict and describe scents.**

  • **[View Live Application](https://your-streamlit-app-link-here.streamlit.app/)** • **[Report a Bug](https://github.com/your-username/ai-that-can-smell/issues)** •
</div>

---

## 📖 Overview
This project simulates an end-to-end "electronic nose." It reads the chemical structure of molecules (SMILES strings), simulates how an array of generic gas sensors would react to those fragments, and uses a trained Random Forest classifier to identify the scent. Finally, it uses the **OpenPOM** (Principal Odor Map) architecture to translate the chemical fragments into human-readable scent descriptors.

It runs locally in minutes on any laptop CPU without requiring dedicated GPU hardware or physical gas sensors.

## 🛠️ Tech Stack
* **Machine Learning:** `scikit-learn` (Random Forest Classifier)
* **Chemical Informatics:** `pubchempy`, `rdkit` (SMILES structural mapping)
* **Neural Architecture:** OpenPOM / DeepChem (simulated for CPU)
* **Frontend:** `streamlit`
* **Data Processing:** `numpy`, `pandas`

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ai-that-can-smell.git](https://github.com/your-username/ai-that-can-smell.git)
   cd ai-that-can-smell
   
Create a virtual environment:

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
🧠 Training & Execution
1. Train the Electronic Nose
Generate synthetic molecular data and train the classifier by running:

Bash
python train_nose.py
(This generates the nose_model.pkl file used by the application).

2. Launch the Web App
Start the Streamlit interface to interact with the pipeline:

Bash
streamlit run app.py
⚠️ Notes on Hardware
The full OpenPOM model requires PyTorch and DeepChem with weights totaling ~5GB. To ensure this project runs instantly on standard laptops or free-tier Streamlit Cloud servers without memory timeouts, the Graph Neural Network integration point in smell_model.py utilizes a probabilistic structural simulator.

🙏 Acknowledgments
The foundational research and dataset structure for this project are based on the OpenPOM framework (MIT License), an open-source rebuild of Google's smell research (Lee et al., A principal odor map unifies diverse tasks in olfactory perception, Science, 2023).


### Important Formatting Notes
1. **Change the URL:** Make sure to replace `[https://your-streamlit-app-link-here.streamlit.app/](https://your-streamlit-app-link-here.streamlit.app/)` with the actual URL Streamlit gave you[cite: 2].
2. **Update the GitHub Links:** Replace `your-username` and `ai-that-can-smell` in the "Report a Bug" link and the `git clone` command with your actual GitHub username and repository name.