# Twitter-Hate-Speech-Detection-end-to-end-using-RNN

## Project Workflow
- constants
- config_enity
- artifact_enity
- components
- pipeline
- app.py

## How to run the application?

#### 1. Create virtual environment
```bash
conda create -n hatevenv
```
#### 2. Activate virtual environment
```bash
conda activate hatevenv
```
#### 3. Download important libraries
```bash
pip install -r requirements.txt
```

# Gcloud link
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe

initiate GCloud

```bash
gcloud init
```

# AWS-CICD-Deployment-with-Github-Actions
## 1. Login to AWS console.
## 2. Create IAM user for deployment

```bash
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

## 3. Create ECR repo to store/save docker image
```bash
- Save the URI : 443370672562.dkr.ecr.ap-south-1.amazonaws.com/project-hate
```
## 4. Create EC2 machine (Ubuntu)
## 5. Open EC2 and Install docker in EC2 Machine:

```bash
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```
## 6. Configure EC2 as self-hosted runner:
```bash
setting>actions>runner>new self hosted runner> choose os> then run command one by one
```
## 7. Setting GitHub Secrects

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO

