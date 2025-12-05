import mlflow
from mlflow.tracking import MlflowClient

from src.logger import logging
from src.utils.main_utils import load_json

def push_model():
    try:
        # ------------------------load eval result------------------------
        eval_result = load_json("artifacts/model_evaluation/evaluation_result.json")
        
        if not eval_result["is_model_accepted"]:
            logging.info("Model was not accepted during evaluation. Exiting pusher.")
            return

        run_id = eval_result["run_id"]
        model_name = eval_result["model_name"]
        model_uri = f"runs:/{run_id}/model"

        mlflow.set_tracking_uri("http://ec2-51-20-74-217.eu-north-1.compute.amazonaws.com:5000")
        client = MlflowClient()

        # ------------------------register current model cause its better------------------------
        logging.info(f"Registering model: {model_name} from run: {run_id}")
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        version = model_details.version
        logging.info(f"Model registered. Version: {version}")

        # ------------------------promotion of model to production------------------------
        logging.info(f"Transitioning version {version} to Production stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True # This automatically archives the OLD production model
        )
        
        logging.info(f"Model {model_name} version {version} is now in Production!")

    except Exception as e:
        logging.error(f"Model pushing failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    push_model()