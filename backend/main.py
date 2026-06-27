import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

# Suppress RDKit warnings
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

# Global variables for model weights
model_weights = {}

@app.on_event("startup")
def startup_event():
    global model_weights
    # Initialize DB
    database.init_db()
    
    # Load model weights from JSON
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Weights JSON not found at {weights_path}. Please export weights first.")
    
    print(f"Loading model weights from {weights_path}...")
    with open(weights_path) as f:
        weights = json.load(f)
        
    model_weights["w0"] = np.array(weights["dense_0_kernel"]) # shape (1024, 256)
    model_weights["b0"] = np.array(weights["dense_0_bias"]) # shape (256,)
    model_weights["scale"] = np.array(weights["bn_1_scale"]) # shape (256,)
    model_weights["offset"] = np.array(weights["bn_1_offset"]) # shape (256,)
    model_weights["w3"] = np.array(weights["dense_3_kernel"]) # shape (256, 64)
    model_weights["b3"] = np.array(weights["dense_3_bias"]) # shape (64,)
    model_weights["w4"] = np.array(weights["dense_4_kernel"]) # shape (64, 1)
    model_weights["b4"] = np.array(weights["dense_4_bias"]) # shape (1,)
    print("Model weights loaded successfully.")

# Input schema
class PredictRequest(BaseModel):
    smiles: str

def run_numpy_inference(fp_array):
    """Runs a forward pass of the neural network using pure NumPy."""
    # Layer 0: Dense
    z0 = np.dot(fp_array, model_weights["w0"]) + model_weights["b0"]
    
    # Layer 1: ReLU + Batch Normalization
    a0 = np.maximum(0, z0) * model_weights["scale"] + model_weights["offset"]
    
    # Layer 3: Dense
    z3 = np.dot(a0, model_weights["w3"]) + model_weights["b3"]
    
    # Activation: ReLU
    a3 = np.maximum(0, z3)
    
    # Layer 4: Dense (Linear)
    y = np.dot(a3, model_weights["w4"]) + model_weights["b4"]
    return float(y[0])

# Endpoints
@app.post("/api/predict")
def predict_ic50(payload: PredictRequest):
    global model_weights
    if not model_weights:
        raise HTTPException(status_code=500, detail="Model weights not loaded yet.")
    
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
        arr = np.zeros((1024,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        
        # Run NumPy inference (predict pIC50)
        pic50_pred = run_numpy_inference(arr)
        
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
            "fingerprint": arr.astype(int).tolist()
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
