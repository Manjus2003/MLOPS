# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "shettymanju2003"
repo_name = "MLOPS"
# # Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/shettymanju2003/MLOPS.mlflow')
# dagshub.init(repo_owner='shettymanju2003', repo_name='MLOPS', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict, max_retries: int = 5):
    """Register the model to the MLflow Model Registry with retry logic."""
    import time
    
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    logging.info(f"Attempting to register model from: {model_uri}")
    
    client = mlflow.tracking.MlflowClient()
    
    # Retry logic for remote tracking server sync
    for attempt in range(max_retries):
        try:
            # Verify the run exists
            run = client.get_run(model_info['run_id'])
            logging.info(f"Run {model_info['run_id']} found with status: {run.info.status}")
            
            # Try to register the model directly (MLflow will handle artifact resolution)
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition the model to "Staging" stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logging.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
            print(f"✓ Model registered successfully: {model_name} v{model_version.version}")
            return model_version
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's an artifact not found error
            if "artifact" in error_msg or "not found" in error_msg or "unable to find" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff: 10, 20, 30, 40 seconds
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    logging.info(f"Artifacts not yet synced. Waiting {wait_time} seconds for DagsHub remote sync...")
                    print(f"⏳ Waiting {wait_time}s for DagsHub to sync artifacts... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logging.error(f'All registration attempts failed after waiting for sync: {e}')
                    raise Exception(f"Failed to register model after {max_retries} attempts ({max_retries * 10}s total wait): {e}. The artifacts may need more time to sync to DagsHub.")
            else:
                # For other errors, don't retry
                logging.error(f'Registration failed with non-sync error: {e}')
                raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        
        # Check if model was already registered during evaluation
        client = mlflow.tracking.MlflowClient()
        try:
            # Try to get the latest version of the model
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            
            if latest_versions:
                # Model exists, find the version from our run
                for mv in latest_versions:
                    if mv.run_id == model_info['run_id']:
                        logging.info(f"Model already registered during evaluation as version {mv.version}")
                        # Transition to Staging
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Staging"
                        )
                        print(f"✓ Model '{model_name}' version {mv.version} transitioned to Staging")
                        logging.info(f"Model {model_name} v{mv.version} transitioned to Staging")
                        return
                
                # If we get here, the latest version doesn't match our run, so register as new version
                logging.info(f"Run {model_info['run_id']} not found in existing versions, registering new version...")
                register_model(model_name, model_info)
            else:
                # No versions exist, register
                logging.info("No existing versions found, registering model...")
                register_model(model_name, model_info)
                
        except Exception as e:
            # If model doesn't exist or any error checking, try to register
            logging.info(f"Could not find existing model registration: {e}. Attempting to register...")
            register_model(model_name, model_info)
            
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

