import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import logging
import tempfile
import json
import mlflow
import mlflow.tracking
import mlflow.artifacts
import sqlite3
import shutil
import os
import io
import boto3
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import requests
from palmerpenguins import load_penguins
from .config import (
    PREPROCESSING_RUN_NAME, PREPROCESSED_DB_POINTER_ARTIFACT_NAME,
    LOCAL_DOWNLOADED_DB_PATH, SELECTED_FEATURES,
    FLIPPER_LENGTH_MM_MIN_ADJUSTED, FLIPPER_LENGTH_MM_MAX_ADJUSTED,
    BILL_LENGTH_MM_MIN_ADJUSTED, BILL_LENGTH_MM_MAX_ADJUSTED,
    BILL_DEPTH_MM_MIN_ADJUSTED, BILL_DEPTH_MM_MAX_ADJUSTED,
    API_KEY, API_BASE_URL, MLFLOW_S3_ENDPOINT_URL,
    AWS_ACCESS_KEY_ID_UI, AWS_SECRET_ACCESS_KEY_UI
)


# --- Configuration & Global variables ---
# Initialize the Dash app first
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
app.title = "Pinguin Klassifikator"
app.logger = logging.getLogger(__name__)
server = app.server  # Expose server for Gunicorn

SPECIES_IMAGES = {
    "Adelie": "Adelie.svg",
    "Gentoo": "Gentoo.svg",
    "Chinstrap": "Chinstrap.svg"
}
DEFAULT_IMAGE = "BW_penguin.svg"

# Input ranges (Min, Max, Step, Default)
INPUT_RANGES = {
    "bill_length_mm": {
        "min": BILL_LENGTH_MM_MIN_ADJUSTED,
        "max": BILL_LENGTH_MM_MAX_ADJUSTED,
        "default": "",
        "label": "Schnabellänge (mm)"
    },
    "bill_depth_mm": {
        "min": BILL_DEPTH_MM_MIN_ADJUSTED,
        "max": BILL_DEPTH_MM_MAX_ADJUSTED,
        "default": "",
        "label": "Schnabeltiefe (mm)"
    },
    "flipper_length_mm": {
        "min": FLIPPER_LENGTH_MM_MIN_ADJUSTED,
        "max": FLIPPER_LENGTH_MM_MAX_ADJUSTED,
        "default": "",
        "label": "Flossenlänge (mm)"
    },
}

# Load data via MLflow S3 pointer or fallback
def load_data_from_s3_or_fallback():
    """Load penguin data from S3 via MLflow or fallback to palmerpenguins dataset."""
    try:
        app.logger.info("Attempting to load data from S3 via MLflow pointer.")
        client = mlflow.tracking.MlflowClient()

        # 1. Find the latest successful preprocessing run
        # Assuming experiment_id '0' (default experiment)
        runs = client.search_runs(
            experiment_ids=['0'],
            filter_string=f"tags.'mlflow.runName' = '{PREPROCESSING_RUN_NAME}' AND status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if not runs:
            raise FileNotFoundError(f"No successful MLflow run found: '{PREPROCESSING_RUN_NAME}'.")
        run_id = runs[0].info.run_id
        app.logger.info(f"Found latest preprocessing run: {run_id}")

        # 2. Download and read the S3 pointer artifact
        database_s3_uri = None
        with tempfile.TemporaryDirectory() as tmpdir:
            client.download_artifacts(
                run_id=run_id,
                path=PREPROCESSED_DB_POINTER_ARTIFACT_NAME,
                dst_path=tmpdir
            )
            pointer_path = Path(tmpdir) / PREPROCESSED_DB_POINTER_ARTIFACT_NAME
            if not pointer_path.exists():
                raise FileNotFoundError("S3 pointer artifact not downloaded.")
            with open(pointer_path, 'r') as f:
                pointer_content = json.load(f)
            database_s3_uri = pointer_content.get("database_s3_uri")
            if not database_s3_uri:
                raise ValueError("S3 URI for database not found in pointer artifact.")
        app.logger.info(f"Database S3 URI: {database_s3_uri}")

        # 3. Download the actual database from S3 to the final target path
        final_db_path = Path(LOCAL_DOWNLOADED_DB_PATH)
        target_download_directory = final_db_path.parent
        target_download_directory.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

        # Robustly remove existing target (file or directory) to ensure a clean download
        if final_db_path.exists():
            app.logger.info(f"Removing existing target at {final_db_path} before download.")
            if final_db_path.is_dir():
                shutil.rmtree(final_db_path)
                app.logger.info(f"Removed existing directory: {final_db_path}")
            elif final_db_path.is_file() or final_db_path.is_symlink():
                final_db_path.unlink()
                app.logger.info(f"Removed existing file/symlink: {final_db_path}")
            else:
                app.logger.warning(f"Existing path {final_db_path} is of an unknown type. Attempting to remove with unlink.")
                try:
                    final_db_path.unlink(missing_ok=True)
                except OSError as oe:
                    app.logger.error(f"Could not remove existing path {final_db_path} of unknown type: {oe}")

        app.logger.info(f"Attempting to download database from {database_s3_uri} to be saved as {final_db_path}. MLflow dst_path: {target_download_directory}")
        mlflow.artifacts.download_artifacts(
            artifact_uri=database_s3_uri, # S3 URI of the database file
            dst_path=str(target_download_directory) # Download into the parent directory
        )

        if not final_db_path.exists() or final_db_path.stat().st_size == 0:
            raise FileNotFoundError(f"Database file {final_db_path} failed to download or is empty.")
        app.logger.info(f"Database downloaded successfully to {final_db_path} (Size: {final_db_path.stat().st_size} bytes).")

        # 4. Load data from the downloaded SQLite database
        conn = sqlite3.connect(final_db_path)
        df = pd.read_sql_query("SELECT * FROM penguins_processed;", conn) # Ensure table name is correct
        conn.close()

        df.dropna(subset=SELECTED_FEATURES + ['species'], inplace=True) # Basic cleaning
        df.reset_index(drop=True, inplace=True)
        app.logger.info(f"Successfully loaded {len(df)} rows from S3-downloaded database: {final_db_path}")
        return df

    except Exception as e:
        app.logger.error(f"Error loading data from S3/MLflow: {e}. Falling back to palmerpenguins dataset.")
        # Fallback logic
        df_fallback = load_penguins()
        cols_to_keep = SELECTED_FEATURES + ['species']  # Add other relevant columns for plot
        cols_to_keep = [col for col in cols_to_keep if col in df_fallback.columns]
        df_fallback = df_fallback[cols_to_keep]
        df_fallback.dropna(subset=SELECTED_FEATURES + ['species'], inplace=True)
        df_fallback.reset_index(drop=True, inplace=True)
        app.logger.info(f"Successfully loaded and preprocessed {len(df_fallback)} rows from palmerpenguins dataset as fallback.")
        return df_fallback

# Now load the data
penguins_df = load_data_from_s3_or_fallback()


app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='plot-trigger-store'),
    dcc.Store(id='last-classified-data-store'), # Store for last classified data
    dbc.Row(
        dbc.Col(html.H1("🐧 Pinguin Klassifikator", className="text-center text-primary my-4"), width=12)
    ),
    dbc.Row([
        dbc.Col(md=4, children=[
            dbc.Card([
                dbc.CardHeader(html.H4("Pinguin-Merkmale eingeben", className="card-title")),
                dbc.CardBody([
                    *[dbc.Form([
                        dbc.Label(INPUT_RANGES[key]["label"], html_for=f"input-{key}"),
                        dbc.Input(
                            id=f"input-{key}",
                            type="text",
                            inputmode="numeric",
                            placeholder=f"Bereich: {INPUT_RANGES[key]['min']} - {INPUT_RANGES[key]['max']}",
                            value=INPUT_RANGES[key]["default"], # Will now be an empty string
                            className="mb-2"
                        )]) for key in INPUT_RANGES.keys()],
                    dbc.Row([
                        dbc.Col(dbc.Button("Klassifiziere Pinguin", id="classify-button", color="primary", className="w-100 mt-3", n_clicks=0), width=6),
                        dbc.Col(dbc.Button("Speichere Pinguin", id="save-penguin-button", color="secondary", className="w-100 mt-3", n_clicks=0, disabled=True), width=6),
                    ], className="g-2"), # g-2 for gutter between columns
                ])
            ], className="shadow-sm mb-4"),
            dbc.Card([
                dbc.CardHeader(html.H4("Klassifizierungsergebnis", className="card-title")),
                dbc.CardBody(
                    dcc.Loading(id="loading-spinner", type="default", children=[
                        html.Div(id="output-classification", className="text-center", children=[
                            html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
                            html.P("Bitte geben Sie die Merkmale des Pinguins ein.")
                        ])
                    ])
                )
            ], className="shadow-sm")
        ]),
        dbc.Col(md=8, children=[
            dbc.Card([
                dbc.CardHeader(html.H4("3D Visualisierung der Pinguin-Daten", className="card-title")),
                dbc.CardBody(dcc.Graph(id="penguin-3d-plot", style={'height': '70vh'}))
            ], className="shadow-sm")
        ])
    ], className="mt-4"),
    html.Footer(
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.P("© Marcel Knauf | 2025 | Pinguin Klassifikations-Demo", className="text-center text-muted small mb-0"),
                        html.P("Datenschutz-Hinweis: Es werden ausschließlich die für den jeweiligen Zweck unbedingt erforderlichen Daten erhoben. Eine dauerhafte Speicherung erfolgt nicht.", className="text-center text-muted small mb-0")
                    ], className="mt-3")
                )
            )
        )
    )
])

@app.callback(
    [Output("output-classification", "children"),
     Output("plot-trigger-store", "data"),
     Output("save-penguin-button", "disabled"),
     Output("last-classified-data-store", "data")],
    [Input("classify-button", "n_clicks")],
    [State(f"input-{key}", "value") for key in INPUT_RANGES.keys()],
    prevent_initial_call=True
)
def classify_penguin_callback(n_clicks, *values):
    """Handle penguin classification button clicks and validate inputs."""
    app.logger.info(f"Classify callback. n_clicks: {n_clicks}")
    store_data_to_send = {'show': False, 'point': None}
    last_classified_data = None # Initialize
    button_disabled_state = True # Default to disabled

    if n_clicks is None or n_clicks == 0:
        initial_html = html.Div([
            html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
            html.P("Bitte geben Sie die Merkmale des Pinguins ein.")
        ])
        return initial_html, store_data_to_send, button_disabled_state, last_classified_data

    processed_values = []
    keys_in_order = list(INPUT_RANGES.keys())
    valid_input = True

    for i, key_name in enumerate(keys_in_order):
        raw_value_from_state = values[i]
        config_for_key = INPUT_RANGES[key_name]
        current_processed_value = None
        iteration_valid = True

        if raw_value_from_state is not None:
            if isinstance(raw_value_from_state, str):
                stripped_value = raw_value_from_state.strip()
                if stripped_value == "":
                    app.logger.warning(f"Input for '{key_name}' is empty.")
                    iteration_valid = False
                else:
                    try:
                        float_val = float(stripped_value)
                        if not (config_for_key["min"] <= float_val <= config_for_key["max"]):
                            app.logger.warning(f"Input for '{key_name}' ('{float_val}') out of range ({config_for_key['min']}-{config_for_key['max']}).")
                            iteration_valid = False
                        else:
                            current_processed_value = float_val
                    except ValueError:
                        app.logger.warning(f"Invalid numeric input for '{key_name}': '{stripped_value}'.")
                        iteration_valid = False
            elif isinstance(raw_value_from_state, (int, float)):
                if not (config_for_key["min"] <= raw_value_from_state <= config_for_key["max"]):
                    app.logger.warning(f"Input for '{key_name}' ('{raw_value_from_state}') out of range ({config_for_key['min']}-{config_for_key['max']}).")
                    iteration_valid = False
                else:
                    current_processed_value = float(raw_value_from_state)
            else:
                app.logger.warning(f"Unexpected type for '{key_name}': {type(raw_value_from_state)}.")
                iteration_valid = False
        else:
            app.logger.warning(f"Input for '{key_name}' is None (missing).")
            iteration_valid = False

        processed_values.append(current_processed_value)
        if not iteration_valid:
            valid_input = False

    app.logger.info(f"Processed values: {processed_values}, Overall Valid input: {valid_input}")

    if not valid_input:
        error_messages = []
        for i, key_name in enumerate(keys_in_order):
            if processed_values[i] is None:
                original_value = values[i]
                if original_value is None or (isinstance(original_value, str) and original_value.strip() == ""):
                    error_messages.append(f"'{INPUT_RANGES[key_name]['label']}': Eingabe fehlt.")
                else:
                    error_messages.append(f"'{INPUT_RANGES[key_name]['label']}': Ungültiger Wert oder außerhalb des Bereichs ({INPUT_RANGES[key_name]['min']}-{INPUT_RANGES[key_name]['max']}).")
        if not error_messages:
             error_messages.append("Ungültige oder fehlende Eingaben. Bitte überprüfen Sie die Werte und Bereiche.")
        output_html = html.Div([
            html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
            html.P("Fehler bei der Eingabe:", style={'color': 'red', 'fontWeight': 'bold'}),
            html.Ul([html.Li(msg, style={'color': 'red'}) for msg in error_messages])
        ])
        return output_html, store_data_to_send, True, None

    feature_values = dict(zip(keys_in_order, processed_values))
    app.logger.info(f"Calling API for prediction with features: {feature_values}")

    try:
        api_key_to_use = os.getenv("API_KEY", API_KEY)
        prediction_result = api_call(
            flipper_length_from_ui=feature_values['flipper_length_mm'],
            bill_length_from_ui=feature_values['bill_length_mm'],
            bill_depth_from_ui=feature_values['bill_depth_mm'],
            api_key_to_use=api_key_to_use
        )

        if isinstance(prediction_result, dict) and "error" in prediction_result:
            error_detail = prediction_result.get('details', prediction_result['error'])
            app.logger.error(f"API call failed: {prediction_result['error']}. Details: {error_detail}")
            output_html = html.Div([
                html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
                html.P("Fehler bei der API-Anfrage:", style={'color': 'red', 'fontWeight': 'bold'}),
                html.P(f"{prediction_result['error']}", style={'color': 'red'}),
                html.P(f"Details: {error_detail}", style={'color': 'red', 'fontSize': 'small'}) if error_detail != prediction_result['error'] and error_detail else ""
            ])
            return output_html, store_data_to_send, True, None

        predicted_species = prediction_result.get("prediction")
        prediction_proba = prediction_result.get("confidence")

        if predicted_species:
            img_url = app.get_asset_url(SPECIES_IMAGES.get(predicted_species, DEFAULT_IMAGE))
            proba_text = f"{prediction_proba*100:.2f}%" if prediction_proba is not None else "N/A"
            output_html = html.Div([
                html.Img(src=img_url, style={"width": "150px", "margin-bottom": "10px"}),
                html.H5(f"Vorhergesagte Spezies: {predicted_species}"),
                html.P(f"Konfidenz: {proba_text}")
            ])
            store_data_to_send = {'show': True, 'point': feature_values} # Use feature_values dict for plot
            button_disabled_state = False
            last_classified_data = {
                "features": feature_values,
                "predicted_species": predicted_species
            }
            app.logger.info(f"Classification via API successful. Predicted: {predicted_species}. Save button enabled.")
        else:
            app.logger.warning(f"API call successful but 'predicted_species' missing in response: {prediction_result}")
            output_html = html.Div([
                html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
                html.P("Klassifizierung via API ergab kein Ergebnis. Bitte versuchen Sie es erneut.", style={'color': 'orange'})
            ])

    except Exception as e:
        app.logger.error(f"Unexpected error during classification process: {e}", exc_info=True)
        output_html = html.Div([
            html.Img(src=app.get_asset_url(DEFAULT_IMAGE), style={"width": "150px", "margin-bottom": "10px", "opacity": "0.3"}),
            html.P(f"Unerwarteter Fehler bei der Klassifizierung: {str(e)}", style={'color': 'red'})
        ])

    return output_html, store_data_to_send, button_disabled_state, last_classified_data


@app.callback(
    [Output("save-penguin-button", "disabled", allow_duplicate=True),
     Output("output-classification", "children", allow_duplicate=True)],
    Input("save-penguin-button", "n_clicks"),
    State("last-classified-data-store", "data"),
    prevent_initial_call=True
)
def save_penguin_callback(n_clicks, classified_data):
    """Save classified penguin data to S3 storage."""
    if n_clicks is None or n_clicks == 0 or not classified_data:
        return True, dash.no_update

    app.logger.info(f"Save penguin callback triggered. Data: {classified_data}")

    features = classified_data.get("features")
    predicted_species = classified_data.get("predicted_species")

    if not features or not predicted_species:
        app.logger.error("Missing features or predicted species in stored data. Cannot save.")
        error_html = html.Div([
            html.P("Fehler: Fehlende Daten zum Speichern. Bitte erneut klassifizieren.", style={'color': 'red'})
        ])
        return True, error_html

    data_to_save = {key: [value] for key, value in features.items()}
    data_to_save["species"] = [predicted_species]

    try:
        df_new_penguin = pd.DataFrame(data_to_save)
        app.logger.info(f"DataFrame for new penguin created: \\n{df_new_penguin.to_string()}")

        csv_buffer = io.StringIO()
        df_new_penguin.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        s3_bucket = os.getenv("DATA_LAKE_BUCKET_UI", "new-penguin-data")
        s3_key = f"incoming_penguin_data/new_penguin_{timestamp}.csv"

        s3_client = boto3.client(
            's3',
            endpoint_url=MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id= AWS_ACCESS_KEY_ID_UI,
            aws_secret_access_key= AWS_SECRET_ACCESS_KEY_UI
        )

        try:
            s3_client.create_bucket(Bucket=s3_bucket)
            app.logger.info(f"Bucket '{s3_bucket}' created successfully.")
        except Exception as bucket_error:
            # Check if it's just because bucket already exists
            error_msg = str(bucket_error).lower()
            if any(keyword in error_msg for keyword in ['already', 'exists', 'owned']):
                app.logger.info(f"Bucket '{s3_bucket}' already exists, continuing...")
            else:
                # If it's a different error, log it but try to continue anyway
                app.logger.warning(f"Bucket creation issue (continuing anyway): {bucket_error}")

        # Upload the file
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_content)
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        app.logger.info(f"New penguin data saved to S3: {s3_uri}")

        # Return success with confirmation message
        success_html = html.Div([
            html.P("✅ Pinguin erfolgreich gespeichert!", style={'color': 'green', 'fontWeight': 'bold'})
        ])
        return True, success_html

    except Exception as e:
        app.logger.error(f"Error during save: {e}", exc_info=True)
        error_message_html = html.Div(html.P(f"Fehler beim Speichern: {str(e)}", style={'color': 'red'}))
        return False, error_message_html

@app.callback(
    Output("penguin-3d-plot", "figure"),
    [Input("plot-trigger-store", "data")]
)
def update_3d_plot(plot_trigger_data):
    """Update 3D scatter plot with penguin data and user input point."""
    app.logger.info(f"update_3d_plot triggered. Data from store: {plot_trigger_data}")
    # Base 3D scatter plot for all penguin data
    fig = px.scatter_3d(
        penguins_df,
        x="bill_length_mm",
        y="bill_depth_mm",
        z="flipper_length_mm",
        color="species",
        symbol="species",
        opacity=0.6,
        hover_data=penguins_df.columns,
        color_discrete_map={"Adelie": "blue", "Gentoo": "green", "Chinstrap": "red"},
        title="Interaktive 3D Ansicht der Pinguin-Daten"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Spezies',
        scene=dict(
            xaxis_title="Schnabellänge (mm)",
            yaxis_title="Schnabeltiefe (mm)",
            zaxis_title="Flossenlänge (mm)"
        )
    )

    if plot_trigger_data and plot_trigger_data.get('show') and plot_trigger_data.get('point'):
        point_to_plot = plot_trigger_data['point'] # This is feature_values dict

        try:
            # feature_values is already a dict like {'bill_length_mm': val, ...}
            current_input_df = pd.DataFrame([point_to_plot])

            fig.add_trace(
                go.Scatter3d(
                    x=current_input_df["bill_length_mm"],
                    y=current_input_df["bill_depth_mm"],
                    z=current_input_df["flipper_length_mm"],
                    mode='markers',
                    marker=dict(size=10, color='black', symbol='diamond'),
                    name='Ihre Eingabe'
                )
            )

            fig.update_layout(scene_annotations=[
                dict(
                    x=current_input_df["bill_length_mm"].iloc[0],
                    y=current_input_df["bill_depth_mm"].iloc[0],
                    z=current_input_df["flipper_length_mm"].iloc[0],
                    text="Ihre Eingabe",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="red",
                    font=dict(color="black", size=12),
                    ax=40,
                    ay=-40
                )
            ])
            app.logger.info(f"Added user input point to 3D plot: {current_input_df.to_dict('records')[0]}")
        except (ValueError, TypeError, KeyError) as e:
            app.logger.error(f"Error processing point data for plotting: {e}. Point data: {point_to_plot}")
            return fig # Return base fig if conversion fails

    return fig

def api_call(flipper_length_from_ui: float, bill_length_from_ui: float, bill_depth_from_ui: float, api_key_to_use: str):
    """Make HTTP request to inference API for penguin classification."""
    app.logger.info(f"API Call with Key: {api_key_to_use} for values: FlipperLength={flipper_length_from_ui}, BillLength={bill_length_from_ui}, BillDepth={bill_depth_from_ui}")

    # Use API_BASE_URL from config
    api_url = f"{API_BASE_URL}/api/predict"

    payload = {
        "flipper_length_mm": flipper_length_from_ui,
        "bill_length_mm": bill_length_from_ui,
        "bill_depth_mm": bill_depth_from_ui
    }
    headers = {
        "X-API-Key": api_key_to_use,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        app.logger.error(f"HTTP error occurred: {http_err} - {response.text}")
        return {"error": f"HTTP error: {response.status_code} {response.reason}", "details": response.text if response else "No response"}
    except requests.exceptions.ConnectionError as conn_err:
        app.logger.error(f"Connection error occurred: {conn_err}")
        return {"error": "Connection error: Could not connect to the API."}
    except requests.exceptions.Timeout as timeout_err:
        app.logger.error(f"Timeout error occurred: {timeout_err}")
        return {"error": "Timeout error: The API request timed out."}
    except requests.exceptions.RequestException as req_err:
        app.logger.error(f"An error occurred during the API request: {req_err}")
        return {"error": "API request error: An unexpected error occurred."}
    except json.JSONDecodeError:
        app.logger.error(f"JSON decode error: Failed to parse response from API. Response: {response.text if response else 'No response'}")
        return {"error": "Invalid response: Could not parse JSON from API."}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
