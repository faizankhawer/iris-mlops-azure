
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Azure ML connection
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="8e1bbb27-dffe-42c8-98ba-e3007fb31500",
    resource_group_name="Gene_new",
    workspace_name="Designer_gene"
)

# Read training job name
with open("last_job_name.txt", "r") as f:
    job_name = f.read().strip()

# Verify job completion
job = ml_client.jobs.get(job_name)

if job.status != "Completed":
    raise Exception(
        f"Training job is not completed. Status: {job.status}"
    )

print("Training job:", job_name)

# Define model
model = Model(
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
    name="iris-random-forest",
    description="Iris Random Forest model tracked using MLflow",
    type=AssetTypes.MLFLOW_MODEL
)

# Register model
registered_model = ml_client.models.create_or_update(model)

print("\nMODEL REGISTERED SUCCESSFULLY!")
print("Model Name:", registered_model.name)
print("Model Version:", registered_model.version)
print("Model ID:", registered_model.id)

