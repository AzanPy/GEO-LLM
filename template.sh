
# .ebextensions
mkdir -p .ebextensions
touch .ebextensions/01_fastapi.config
touch .ebextensions/02_security.config

# .platform hooks
mkdir -p .platform/httpd/conf.d
touch .platform/httpd/conf.d/ssl_rewrite.conf

# src directory
mkdir -p src
touch src/__init__.py
touch src/embedder.py
touch src/indexer.py
touch src/retriever.py
touch src/query_analyzer.py
touch src/rag_chain.py
touch src/config.py

# templates
mkdir -p templates
touch templates/chat.html

# static
mkdir -p static
touch static/style.css

# data (local only)
mkdir -p data/studies
mkdir -p data/samples

# Root files
touch app.py
touch requirements.txt
touch Procfile
touch Dockerfile
touch docker-compose.yml
touch .env.example

echo "Project structure created successfully!"