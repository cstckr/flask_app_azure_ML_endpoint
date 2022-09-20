# Flask test app
This is a test flask app ready to be deployed as an Azure web app (via an docker image).
After log-in, the user can upload an image. The app returns a prediction of what digit is shown in the uploaded image.

# Deploy to Azure

- Local training of model: 
    - Run "train_model.py" to train the TensorFlow model.
- Model deployment via Azure REST endpoint:
    - Run "deploy_azure_endpoint.py" to create the Azure REST endpoint.
- Create user database in Azure:
    - See https://github.com/cstckr/another_dockerized_flask_app
- Create docker image and push in to the Azure container registry:
    - See https://github.com/cstckr/another_dockerized_flask_app
- Deploy app:
    - See https://github.com/cstckr/another_dockerized_flask_app