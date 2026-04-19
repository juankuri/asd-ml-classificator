#!/usr/bin/env bash
set -e
python3 -m venv venv
# activar manualmente: source venv/bin/activate
venv/bin/pip install -r requirements.txt

echo "Entorno virtual creado y dependencias instaladas. Usa: source venv/bin/activate"