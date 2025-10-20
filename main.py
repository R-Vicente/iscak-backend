from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import sys
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from isca_k.iscak_core import ISCAkCore

app = FastAPI(title="ISCA-k Imputation API")

# CORS - permite requests do GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://r-vicente.github.io"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                    "rows": len(df),
                    "columns": len(df.columns)
                }
            )
        
        # Inicializa ISCA-k
        imputer = ISCAkCore(
            min_friends=request.min_friends,
            max_friends=request.max_friends,
            mi_neighbors=request.mi_neighbors,
            n_jobs=-1,
            verbose=False,  # Desativa prints para API
            max_cycles=request.max_cycles,
            categorical_threshold=request.categorical_threshold
        )
        
        # Imputa (non-interactive mode para API)
        df_imputed = imputer.impute(
            df,
            force_categorical=None,
            force_ordinal=None,
            interactive=False,  # CRITICAL: Não pode ser interativo na API
            column_types_config=None
        )
        
        # Converte de volta para CSV
        output = io.StringIO()
        df_imputed.to_csv(output, index=False)
        csv_imputed = output.getvalue()
        
        # Estatísticas
        stats = imputer.execution_stats
        stats['rows'] = len(df)
        stats['columns'] = len(df.columns)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
