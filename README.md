# 🧠 Autism ML (AQ-10 Classifier)

Sistema de **Machine Learning para clasificación de autismo (ASD)** basado en datos de screening (cuestionario AQ-10).

> ⚠️ **Importante:** Este modelo no realiza diagnóstico clínico. Su propósito es apoyar la **detección temprana**.

---

# 📁 Estructura del proyecto

```
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
├── models/
│   ├── final_model.pkl
│   ├── columns.pkl
│   ├── threshold.pkl
│   └── model_name.pkl
├── outputs/
│   └── roc_comparison.png
├── app.py
└── main.py
```

---

# ⚙️ Tecnologías utilizadas

* Python 3
* pandas
* scikit-learn
* joblib
* matplotlib
* streamlit (interfaz opcional)

---

# 🚀 Instalación rápida (Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# ▶️ Ejecución del pipeline

```bash
python main.py
```

Esto realiza automáticamente:

1. Carga y limpieza de datos
2. Preprocesamiento (encoding + imputación)
3. Entrenamiento de modelos
4. Evaluación (Accuracy, Recall, ROC-AUC)
5. Comparación de modelos
6. Selección del mejor modelo
7. Guardado de artefactos

---

# 🧠 Modelos utilizados

* Logistic Regression
* Random Forest
* Gradient Boosting

---

# 📊 Métricas evaluadas

* Accuracy
* Precision
* Recall (prioritaria en este proyecto)
* F1-score
* ROC-AUC

---

# ⚖️ Consideraciones importantes

## 🔹 Desbalance de clases

Se utiliza:

```python
class_weight="balanced"
```

Para mejorar la detección de la clase minoritaria (ASD).

---

## 🔹 Ajuste de threshold

Se modifica el threshold por defecto:

```text
0.5 → 0.3
```

Esto permite:

* Detectar más casos de autismo (↑ recall)
* A costa de más falsos positivos

---

# 📦 Artefactos generados

| Archivo              | Descripción                                |
| -------------------- | ------------------------------------------ |
| `final_model.pkl`    | Modelo entrenado                           |
| `columns.pkl`        | Estructura de datos usada en entrenamiento |
| `threshold.pkl`      | Umbral de decisión                         |
| `model_name.pkl`     | Nombre del mejor modelo                    |
| `roc_comparison.png` | Gráfica comparativa de modelos             |

---

# 🖥️ Uso del modelo (predicción)


## Interfaz web

```bash
streamlit run app.py
```

Permite:

* Ingresar datos de usuario
* Obtener predicción en tiempo real
* Visualizar probabilidad

---

# 🔄 Flujo del sistema

```
Datos → Preprocesamiento → Modelo → Probabilidad → Clasificación
```

---

# 📥 Formato del dataset

El dataset debe contener:

* Variables AQ-10 (`A1_Score` a `A10_Score`)
* Variables demográficas (edad, género, etc.)
* Variable objetivo:

```text
Class/ASD → 0 (No), 1 (Sí)
```

---

# 🧪 Preprocesamiento automático

`src/preprocessing.py` realiza:

* Conversión de `yes/no` → `1/0`
* One-Hot Encoding para variables categóricas
* Manejo de valores faltantes

---

# ⚠️ Limitaciones

* Dataset basado en cuestionario (no clínico)
* Posible sesgo en respuestas
* No sustituye diagnóstico médico
* Dependiente de calidad de datos

---

# 🎯 Decisión del modelo

* Mejor modelo global: **Gradient Boosting**
* Mejor enfoque clínico: **Logistic Regression + threshold bajo**

---

# 💡 Justificación clave

> Se prioriza el **recall** para minimizar falsos negativos, dado el impacto clínico de no detectar un caso de autismo.

---

# 📚 Conceptos clave

* Screening
* Clasificación binaria
* One-Hot Encoding
* Imputación
* ROC-AUC
* Threshold tuning

---

# 🧩 Futuras mejoras

* Uso de pipelines completos (`sklearn.pipeline`)
* Validación cruzada
* Optimización de hiperparámetros
* Mejor interfaz de usuario
* Integración como API

