import streamlit as st
from pricePrediction.predict.predict import GraphPricePredictor
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd

st.title("CoPriNet: Molecule Price Predictions")

with st.spinner("Loading model..."):
    n_cpus = 0
    n_gpus = 0
    predictor = GraphPricePredictor(n_cpus=n_cpus, n_gpus=n_gpus)


@st.cache
def get_predictions(smiles: List[str], convert_to_grams: Optional[bool]=False)->List[float]:
    preds = list(predictor.yieldPredictions(smiles))
    
    if convert_to_grams:
        from rdkit import Chem

        def convert_pred(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return np.nan
            mw = Chem.Descriptors.ExactMolWt(mol)
            price = np.log(np.exp(pred) * 1000 / mw)
            return price

        preds = [convert_pred(smi, pred) for smi, pred in zip(smiles, preds)]
    # if args.output_file is None:
    #     for smi, pred in zip(smiles_list, preds):
    #         print("%s\t%.4f" % (smi, pred))
    # else:
    #     df[coprinet_colname] = list(preds)
    #     df.to_csv(args.output_file, index=False)
    return preds


st.write(
    "Enter a SMILES string to get a price prediction. You can also enter a list of comma separated strings or upload a CSV with the column SMILES to get several predictions."
)

upload_csv = st.checkbox("Upload CSV")
# Upload CSV
sample_smiles = None
if upload_csv:
    csv = st.file_uploader("Upload CSV", type="csv")
    if csv is not None:
        df =  pd.read_csv(csv)
        sample_smiles = ",".join(df["SMILES"].tolist())

if sample_smiles is None:
    sample_smiles= "COc1ccc(C(=O)/C=C(/O)c2ccc3c(c2)OCO3)c(OC)c1,C=C(CC(=O)[C@H](C)CCC[C@]1(CO)O[C@H]1CC/C(C)=C/COC(C)=O)C(C)C"
# SMILES input
smiles = st.text_input("Enter SMILES strings", value=sample_smiles)
convert_to_grams = st.checkbox("Convert to grams", value=False)


# Get and dislay predictions
if smiles:
    smiles = smiles.split(",")
    with st.spinner("Predicting..."):
        preds = get_predictions(smiles, convert_to_grams)
    unit = "g" if convert_to_grams else "mmol"
    img = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(smi) for smi in smiles],
        legends=[f"${pred:.02f}/{unit}" for pred in preds],
        molsPerRow=2,
        subImgSize=(400,250)
    )
    st.image(img)

    df = pd.DataFrame({"SMILES": smiles, f"Price ($/{unit})": preds})
    st.download_button("Download predictions", data=df.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime="text/csv")

st.markdown("This app is based on the paper by Sanchez-Garcia et al: [CoPriNet: A Graph Neural Network for Predicting the Price of Molecules](https://chemrxiv.org/engage/chemrxiv/article-details/62bd9cf1d66f68a0b1b7bb5c).")
