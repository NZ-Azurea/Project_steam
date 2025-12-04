Steam Project: Analysis and Generation

Welcome to the Steam data analysis and generation project repository. This project is configured for easy deployment using Docker.

Quick Start

To ensure all large files are downloaded correctly, you must have Git Large File Storage (LFS) installed.

1. Prerequisites: Install Git LFS

Install Git LFS and initialize it:

```bash
git lfs install
```

2. Clone the Repository

Clone the repository:

``` bash
git clone https://github.com/NZ-Azurea/Project_steam.git
cd Project_steam
```

Execution via Docker (Recommended Method)

This method uses docker compose to build and run the application in a single step.

1. Build the Container

Build the Docker image.

``` bash
docker compose build --no-cache
```

2. Launch the Application

``` bash
docker compose up -d
```

Manual Execution (Troubleshooting)

If running via docker compose up encounters issues or to retrain the models.

1. Access the Container

Enter the Bash shell of the application container:

``` bash
docker exec -it steam_app bash
```

2. Run the Application

Once inside the container, grant execution rights and run the application with Uvicorn:

``` bash
chmod +x /start.sh
uv run uvicorn start.sh
```

To see the result. Go to yoyr web-browser and write:

```bash
http://localhost:8501/
```









