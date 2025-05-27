# Training Container

Hier passiert das ML Training. Lädt Pinguin-Daten, trainiert sklearn Model, speichert alles in MLflow.

## Was macht es?
1. Lädt Palmer Penguins Dataset
2. Data Preprocessing
3. Trainiert Logistic Regression
4. Speichert Model + Artifacts in MLflow/S3

Läuft automatisch wenn docker-compose startet. Muss erfolgreich sein bevor UI startet.
