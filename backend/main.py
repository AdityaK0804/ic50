import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

# Suppress RDKit and Tensorflow warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from backend import database
except ImportError:
    import database

# Initialize FastAPI app
app = FastAPI(
    title="IC50 FORGE API",
    description="Backend API and static server for IC50 FORGE drug potency predictor",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to load model and database
model = None

@app.on_event("startup")
def startup_event():
    global model
    # Initialize DB
    database.init_db()
    
    # Load Keras model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic50_model.h5")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Saved model not found at {model_path}. Please train the model first.")
    
    print(f"Loading Keras model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    print("Keras model loaded successfully.")

# Input schema
class PredictRequest(BaseModel):
    smiles: str

# Endpoints
@app.post("/api/predict")
def predict_ic50(payload: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
    
    smiles = payload.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string is empty.")
    
    try:
        # Validate SMILES using RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string. RDKit failed to parse molecule.")
        
        # Generate 1024-bit Morgan Fingerprint, radius 2
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1, 1024), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr[0])
        
        # Run Keras Model prediction
        # Use predict_on_batch or predict to output pIC50
        pic50_pred = float(model.predict(arr, verbose=0)[0][0])
        
        # Convert pIC50 back to micromolar (µM)
        # formula: pIC50 = -log10(EC50 / 1e6) -> EC50 (Molar) = 10^(-pIC50)
        # EC50 (µM) = EC50 (Molar) * 1e6 = 10^(6 - pIC50)
        ic50_um_pred = float(10 ** (6 - pic50_pred))
        
        # Log to Database
        database.log_prediction(smiles, pic50_pred, ic50_um_pred)
        
        # Return response
        return {
            "smiles": smiles,
            "pIC50": pic50_pred,
            "ic50_um": ic50_um_pred,
            "fingerprint": arr[0].astype(int).tolist()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/logs")
def get_prediction_logs():
    try:
        logs = database.get_logs(limit=100)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")

# Mount the static files from the React frontend export (located in frontend/out)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "out")

if os.path.exists(STATIC_DIR):
    print(f"Mounting static files from {STATIC_DIR}")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"Warning: Static files directory not found at {STATIC_DIR}. Make sure you built the frontend.")
