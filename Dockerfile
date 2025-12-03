# ÉTAPE 1: Image de base officielle et légère
# Remplacer 'node:20-alpine' par 'python:3.12-alpine' pour une image Python officielle et minimale.
FROM python:3.13-rc-slim

RUN pip install --no-cache-dir uv

# ÉTAPE 2: Définir le répertoire de travail dans le conteneur
WORKDIR /app

# ÉTAPE 3: Copier le fichier de dépendances Python uniquement
# Cela permet à Docker de mettre en cache cette étape si le fichier requirements.txt ne change pas.
COPY requirements.txt ./

# ÉTAPE 4: Installer les dépendances Python
# Utiliser 'pip install' au lieu de 'npm install'. Le flag --no-cache-dir réduit la taille de l'image.
RUN uv pip install --system --no-cache-dir -r requirements.txt

# ÉTAPE 5: Copier le reste du code de l'application
# Le point '.' final indique de copier le contenu du répertoire de contexte vers /app dans le conteneur.
COPY . .

# ÉTAPE 6: Exposer le port de l'application
# Si votre application Python (par exemple, Flask/Django) écoute sur un port spécifique.
EXPOSE 5000 

# ÉTAPE 7: Commande de démarrage

# Remplacez 'app.py' par le nom de votre fichier d'entrée principal.
RUN chmod +x start.sh
CMD ["tail", "-f", "/dev/null"]