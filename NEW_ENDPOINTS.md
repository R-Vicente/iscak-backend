# Novos Endpoints de Imputação

Foram adicionados 3 novos endpoints ao backend, seguindo o mesmo padrão do ISCA-k.

## Endpoints

### 1. kNN Imputation
```
POST /impute/knn
```

**Request Body:**
```json
{
  "csv_data": "col1,col2,col3\n1.0,2.0,\n3.0,,5.0",
  "k": 5
}
```

**Parâmetros:**
- `csv_data` (string, required): Dados em formato CSV
- `k` (int, optional, default=5): Número de vizinhos

---

### 2. MICE Imputation
```
POST /impute/mice
```

**Request Body:**
```json
{
  "csv_data": "col1,col2,col3\n1.0,2.0,\n3.0,,5.0",
  "max_iter": 10,
  "random_state": 42
}
```

**Parâmetros:**
- `csv_data` (string, required): Dados em formato CSV
- `max_iter` (int, optional, default=10): Número máximo de iterações
- `random_state` (int, optional, default=42): Seed para reprodutibilidade

---

### 3. MissForest Imputation
```
POST /impute/missforest
```

**Request Body:**
```json
{
  "csv_data": "col1,col2,col3\n1.0,2.0,\n3.0,,5.0",
  "n_estimators": 10,
  "max_iter": 10,
  "random_state": 42
}
```

**Parâmetros:**
- `csv_data` (string, required): Dados em formato CSV
- `n_estimators` (int, optional, default=10): Número de árvores no Random Forest
- `max_iter` (int, optional, default=10): Número máximo de iterações
- `random_state` (int, optional, default=42): Seed para reprodutibilidade

---

## Response Format

Todos os endpoints retornam o mesmo formato:

```json
{
  "csv_data": "col1,col2,col3\n1.0,2.0,3.5\n3.0,4.2,5.0",
  "success": true,
  "message": "kNN imputation completed successfully",
  "stats": {
    "initial_missing": 2,
    "final_missing": 0,
    "rows": 2,
    "columns": 3,
    "method": "kNN",
    "k": 5
  }
}
```

---

## Funcionalidades

### Deteção Automática de Tipos

Os métodos detetam automaticamente o tipo de dados:

1. **Numérico Puro** (IRIS, WINE, Diabetes, SONAR)
   - kNN: Standardiza → Imputa → Inverse transform
   - MICE/MissForest: Usa sklearn diretamente

2. **Misto** (Titanic, Credit)
   - kNN: Label encode → One-hot → Standardiza numéricas → KNN → Reverte
   - MICE/MissForest: Label encode → Imputa → Round/clip → Reverte

3. **Categórico Puro** (Mushroom)
   - kNN: Label encode → One-hot → KNN (sem standardização) → Reverte
   - MICE: IterativeImputer
   - MissForest: RandomForestClassifier (não Regressor!)

---

## Como Testar Localmente

### 1. Instalar dependências
```bash
cd iscak-backend
pip install -r requirements.txt
```

### 2. Executar servidor
```bash
python main.py
```

### 3. Testar endpoint (exemplo com curl)
```bash
curl -X POST "http://localhost:8000/impute/knn" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_data": "col1,col2,col3\n1.0,2.0,\n3.0,,5.0",
    "k": 5
  }'
```

---

## Próximos Passos

1. ✅ Backend criado com 3 endpoints
2. ⏳ Fazer deploy no Render (ou onde tens o ISCA-k)
3. ⏳ Atualizar frontend para chamar `/impute/knn`, `/impute/mice`, `/impute/missforest`
4. ⏳ Testar com dados reais

---

## Ficheiros Criados/Modificados

- **`imputation_methods.py`** (NOVO): Implementações de kNN, MICE, MissForest
- **`main.py`** (MODIFICADO): Adicionados 3 novos endpoints
