# Autism ML (AQ-10 classifier)

Estructura del proyecto y pasos rápidos para configurar un entorno virtual y ejecutar el pipeline mínimo.

Estructura esperada:

```
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
├── models/
│   └── model.pkl
├── outputs/
│   └── metrics.txt
└── main.py
```

Pasos rápidos (Linux):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --train data/train.csv --test data/test.csv
```

Notas:
- `data/*.csv` debe contener una columna `label` con 0/1 y las columnas de respuestas AQ-10.
- `src/preprocessing.py` intenta convertir respuestas `yes/no` a 1/0 automáticamente.
