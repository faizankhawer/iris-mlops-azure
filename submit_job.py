from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Connect to Azure ML
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="8e1bbb27-dffe-42c8-98ba-e3007fb31500",
    resource_group_name="Gene_new",
    workspace_name="Designer_gene"
)

# Create environment
env = Environment(
    name="iris-mlflow-environment",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
)

# Create training job
job = command(
    code="./src",
    command="python train.py",
    environment=env,
    compute='faizankhawer1',
    display_name="iris-random-forest-mlflow",
    experiment_name="iris-mlflow-experiment",
    description="Iris classification with MLflow"
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)

# Save job name for registration
with open("last_job_name.txt", "w") as f:
    f.write(returned_job.name)

print("\nTraining job submitted!")
print("Job Name:", returned_job.name)
print("Studio URL:", returned_job.studio_url)

print("\nSave this job name for model registration.")
