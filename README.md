# Pinguin Klassifikator - ML Systems Design

Mein Projekt für Machine Learning Systems Design. Ein ML-System das Pinguinarten klassifiziert.

## Komponenten
- `training_container/`: Training Pipeline mit MLflow
- `ui_inference_container/`: Web UI + API für Predictions
- `docker-compose.yml`: Startet alles zusammen

## Wie starten?
```bash
docker-compose up --build -d
```

Dann:
- UI: http://localhost:8050
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001

Das wars!
