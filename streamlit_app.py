import streamlit as st
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BreastCanAI",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 BreastCanAI")
st.markdown(
    "### AI-based ligand screening for breast cancer drug discovery"
)

st.divider()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/model.joblib")

model = load_model()

# ---------------- USER INPUT ----------------
smiles = st.text_input(
    "Enter ligand SMILES string",
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
)

if smiles:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES string")
    else:
        # ---------------- DESCRIPTORS ----------------
        descriptors = {
            "MolWt": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotB": Descriptors.NumRotatableBonds(mol),
            "NumRings": rdMolDescriptors.CalcNumRings(mol),
            "QED": QED.qed(mol)
        }

        df = pd.DataFrame([descriptors])

        st.subheader("📊 Calculated Molecular Descriptors")
        st.dataframe(df, use_container_width=True)

        # ---------------- PREDICTION ----------------
        if st.button("Predict Breast Cancer Therapeutic Potential"):
            try:
                probability = model.predict_proba(df)[0][1] * 100

                st.subheader("🧠 AI Prediction Result")
                st.metric(
                    "Predicted Therapeutic Probability",
                    f"{probability:.2f} %"
                )
                st.progress(probability / 100)

                if probability >= 70:
                    st.success("✅ High potential candidate")
                elif probability >= 40:
                    st.warning("⚠️ Moderate potential – optimization suggested")
                else:
                    st.error("❌ Low potential candidate")

                st.info(
                    "⚠️ This is an AI-assisted prediction and **not a clinical decision tool**."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
