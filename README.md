# GEO-LLM

# How to run this project using conda (Recommended)
### STEPS:

1. Clone this repository
```bash
 git clone https://github.com/AzanPy/GEO-LLM
```
2. Create a virtual environment
```bash
conda create -n geo-llm python=3.11 -y
```
3. Activate the virtual environment
```bash
conda activate geo-llm
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
QDRANT_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
QDRANT_URL = "https://your-cluster.cloud.qdrant.io"
```



```bash
# Finally run the following command
uvicorn app:app --reload --port 8000
```

Now,
```bash
open up localhost:8000
```
# How to run this project using Docker
### STEPS:

1. Clone this repository
```bash
 git clone https://github.com/AzanPy/GEO-LLM
```
2. Create a .env file in the root directory and add your credentials
```bash
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

```
3. Build the docker image
```bash
docker build -t geo-llm .
```
4. Run the docker image
```bash
docker run -p 8000:8000 --env-file .env geo-llm
```
5. Open your browser 

```bash

http://localhost:8000

```
# How to run this project using Docker Compose (Optional)
### STEPS:

1. Build and start the container
```bash
docker compose up --build
```
2. Run in detached mode
```bash
docker compose up -d
```
3. Stop the container
```bash
docker compose down
```
4. Open your browser 

```bash

http://localhost:8000/health

```
5. ⚠️ Make sure Docker is installed

```bash
docker --version
```


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

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

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - EC2_HOST
   - EC2_SSH_KEY
   - EC2_USER
   - GROQ_API_KEY
   - QDRANT_API_KEY
   - QDRANT_URL