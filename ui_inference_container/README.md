# UI & Inference Container

Dash Web-App + Flask API für Pinguin-Klassifikation.

## Features
- 3D Plot der Pinguin-Daten
- Input-Formular für Merkmale (Schnabel, Flosse)
- Live Prediction via API
- Speichern neuer Daten in S3

## Ports
- 8050: Dash UI
- 8080: Flask API (intern)

Lädt trainiertes Model aus MLflow und macht Predictions. UI ruft interne API auf.
