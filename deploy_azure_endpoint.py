from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import (
    Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment,
    CodeConfiguration)
from credentials.endpoint import (
    subscription_id, resource_group_name, workspace_name, target_name,
    online_endpoint_name)


ml_client = MLClient(credential=DefaultAzureCredential(),
                     subscription_id=subscription_id,
                     resource_group_name=resource_group_name,
                     workspace_name=workspace_name)

# Target
try:
    ml_client.compute.get(name=target_name)
    print("Found existing target")
except Exception:
    compute = AmlCompute(
        name=target_name, size="STANDARD_DS3_V2", min_instances=0,
        max_instances=1)
    ml_client.compute.begin_create_or_update(compute)
    print("Created target...")


# Registering model
model1 = Model(
    path="./model",
    type="custom_model",  # ModelType.CUSTOM,
    name="cstckr-model-2022-09-12",
    description="Tensorflow model created from local file.")
ml_client.models.create_or_update(model1)


# Deployment
online_endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint",
    auth_mode="key")

ml_client.begin_create_or_update(online_endpoint)

myenv = Environment(
    conda_file="./yamls/cstckr_env1.yml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest")

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model1,
    environment=myenv,
    code_configuration=CodeConfiguration(
        code="./scoring", scoring_script="score.py"),
    instance_type="Standard_F2s_v2",
    instance_count=1)

online_endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(online_endpoint)
