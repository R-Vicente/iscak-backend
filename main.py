from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import sys
import uvicorn
import logging
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

# Desativar logs de acesso detalhados
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from isca_k.iscak_core import ISCAkCore
from imputation_methods import impute_knn, impute_mice, impute_missforest

app = FastAPI(title="ISCA-k Imputation API")

# CORS - permite requests do GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://r-vicente.github.io"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para não logar IPs
@app.middleware("http")
async def remove_ip_from_logs(request: Request, call_next):
    response = await call_next(request)
    return response

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class ImputationRequest(BaseModel):
    csv_data: str
    min_friends: int = 3
    max_friends: int = 15
    mi_neighbors: int = 3
    max_cycles: int = 3
    categorical_threshold: int = 10

class ImputationResponse(BaseModel):
    csv_data: str
    success: bool
    message: str
    stats: dict = None

class KNNRequest(BaseModel):
    csv_data: str
    k: int = 5

class MICERequest(BaseModel):
    csv_data: str
    max_iter: int = 10
    random_state: int = 42

class MissForestRequest(BaseModel):
    csv_data: str
    n_estimators: int = 10
    max_iter: int = 10
    random_state: int = 42

@app.get("/")
def read_root():
    return {
        "message": "ISCA-k Imputation API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/impute", response_model=ImputationResponse)
async def impute_data(request: ImputationRequest):
    try:
        # Parse CSV
        df = pd.read_csv(io.StringIO(request.csv_data))
        
        # Verifica se há missing values
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            return ImputationResponse(
                csv_data=request.csv_data,
                success=True,
                message="No missing values detected",
                stats={
                    "initial_missing": 0,
                    "final_missing": 0,
                    "rows": int(len(df)),  
                    "columns": int(len(df.columns))  
                }
            )
        
        # Inicializa ISCA-k
        imputer = ISCAkCore(
            min_friends=request.min_friends,
            max_friends=request.max_friends,
            mi_neighbors=request.mi_neighbors,
            n_jobs=-1,
            verbose=False,
            max_cycles=request.max_cycles,
            categorical_threshold=request.categorical_threshold
        )
        
        # Imputa
        df_imputed = imputer.impute(
            df,
            force_categorical=None,
            force_ordinal=None,
            interactive=False,
            column_types_config=None
        )
        
        # Converte de volta para CSV
        output = io.StringIO()
        df_imputed.to_csv(output, index=False)
        csv_imputed = output.getvalue()
        
        # Estatísticas - CONVERTE NUMPY TYPES 
        stats = convert_numpy_types(imputer.execution_stats)
        stats['rows'] = int(len(df))
        stats['columns'] = int(len(df.columns))
        
        return ImputationResponse(
            csv_data=csv_imputed,
            success=True,
            message="Imputation completed successfully",
            stats=stats
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Imputation failed: {str(e)}"
        )

@app.post("/impute/knn", response_model=ImputationResponse)
async def impute_knn_endpoint(request: KNNRequest):
    try:
        # Parse CSV
        df = pd.read_csv(io.StringIO(request.csv_data))

        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            return ImputationResponse(
                csv_data=request.csv_data,
                success=True,
                message="No missing values detected",
                stats={
                    "initial_missing": 0,
                    "final_missing": 0,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns))
                }
            )

        # Impute using kNN
        df_imputed = impute_knn(df, k=request.k)

        # Convert back to CSV
        output = io.StringIO()
        df_imputed.to_csv(output, index=False)
        csv_imputed = output.getvalue()

        # Stats
        stats = {
            'initial_missing': int(missing_count),
            'final_missing': int(df_imputed.isna().sum().sum()),
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'method': 'kNN',
            'k': request.k
        }

        return ImputationResponse(
            csv_data=csv_imputed,
            success=True,
            message="kNN imputation completed successfully",
            stats=stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"kNN imputation failed: {str(e)}"
        )

@app.post("/impute/mice", response_model=ImputationResponse)
async def impute_mice_endpoint(request: MICERequest):
    try:
        # Parse CSV
        df = pd.read_csv(io.StringIO(request.csv_data))

        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            return ImputationResponse(
                csv_data=request.csv_data,
                success=True,
                message="No missing values detected",
                stats={
                    "initial_missing": 0,
                    "final_missing": 0,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns))
                }
            )

        # Impute using MICE
        df_imputed = impute_mice(
            df,
            max_iter=request.max_iter,
            random_state=request.random_state
        )

        # Convert back to CSV
        output = io.StringIO()
        df_imputed.to_csv(output, index=False)
        csv_imputed = output.getvalue()

        # Stats
        stats = {
            'initial_missing': int(missing_count),
            'final_missing': int(df_imputed.isna().sum().sum()),
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'method': 'MICE',
            'max_iter': request.max_iter
        }

        return ImputationResponse(
            csv_data=csv_imputed,
            success=True,
            message="MICE imputation completed successfully",
            stats=stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MICE imputation failed: {str(e)}"
        )

@app.post("/impute/missforest", response_model=ImputationResponse)
async def impute_missforest_endpoint(request: MissForestRequest):
    try:
        # Parse CSV
        df = pd.read_csv(io.StringIO(request.csv_data))

        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            return ImputationResponse(
                csv_data=request.csv_data,
                success=True,
                message="No missing values detected",
                stats={
                    "initial_missing": 0,
                    "final_missing": 0,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns))
                }
            )

        # Impute using MissForest
        df_imputed = impute_missforest(
            df,
            n_estimators=request.n_estimators,
            max_iter=request.max_iter,
            random_state=request.random_state
        )

        # Convert back to CSV
        output = io.StringIO()
        df_imputed.to_csv(output, index=False)
        csv_imputed = output.getvalue()

        # Stats
        stats = {
            'initial_missing': int(missing_count),
            'final_missing': int(df_imputed.isna().sum().sum()),
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'method': 'MissForest',
            'n_estimators': request.n_estimators,
            'max_iter': request.max_iter
        }

        return ImputationResponse(
            csv_data=csv_imputed,
            success=True,
            message="MissForest imputation completed successfully",
            stats=stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MissForest imputation failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"  # Reduz verbosity
    )
