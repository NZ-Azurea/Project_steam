## 📄 Project Architecture Documentation

### 1. Global Architecture Diagram

**Diagram:** 
![Global architecture](assets/Diagram_global_archi.png)

This diagram illustrates the **overall workflow and component interaction** of the project.

The **API (Application Programming Interface)** acts as the central hub, mediating communication between the front-end user interface and the data/model components.

* **Front-End / User Interface:** This component is the **Streamlit Application**, which is accessed by the end-users. It sends requests to the API for data display and processing.
* **The API (Central Component):** This component processes requests from the Streamlit UI. It interacts with both the **MongoDB Database** (for data retrieval and storage) and the **Machine Learning (ML) Models** (for predictions or generation tasks).
* **Data Layer:** The **MongoDB** database stores the project's persistent data (likely Steam games, reviews, and user information).
* **Modeling Layer:** This layer includes the **ML Models** (e.g., NLGCL and GenSar, as suggested by the setup script). The API queries these models to fulfill the Streamlit application's needs (e.g., recommendations, content generation).

### 2. Docker Architecture Diagram

**Diagram:** 
![Docker architecture](assets/Diagram_docker_archi.png)

This diagram shows how the entire application is containerized using Docker, focusing on isolation and deployment.

It highlights that each major component runs within its own container, managed by Docker Compose (implied by the multi-container setup):

* **Streamlit Container:** Hosts the user interface.
* **API Container:** Hosts the central application logic (e.g., Uvicorn server).
* **MongoDB Container:** Hosts the database instance.

This containerization approach ensures that the application is **portable**, **scalable**, and that dependencies for each service are kept isolated. Communication between these components happens over the Docker network.

### 3. The `entrypoint.sh` Function Diagram

**Diagram:**
![The entrypoint.sh function](assets/Diagram_entrypoint_sh.png)

This diagram likely describes the **initial execution flow** when a Docker container (probably the API or a core service) is started.

The `entrypoint.sh` script defines the primary command that runs when the container first launches. Its typical purpose is to perform essential initialization tasks before starting the main application process.

Common actions within an `entrypoint.sh` include:
* Waiting for the database (MongoDB) to be ready.
* Running database migrations or setup scripts.
* Setting up environment variables.
* Finally, executing the `start.sh` script or the main application command.

### 4. The `start.sh` Function Diagram

**Diagram:** 
![The start.sh function](assets/Diagram_start_sh.png)

This diagram illustrates the **main execution phase** of a core service (likely replacing the steps in the initial Bash bootstrap script for a production/containerized environment).

The `start.sh` script is responsible for initiating the application process itself after the necessary environment has been prepared by `entrypoint.sh`.

It typically includes the commands to:
* Run the Python API server (e.g., `uvicorn API_DB:app`).
* In a multi-service container (less common but possible), it could also launch the Streamlit app.
* It ensures the core services of that specific container are running and listening for connections.

---

**Summary of the Workflow:** The Dockerized architecture enables an efficient deployment where `entrypoint.sh` handles setup and dependency checks, and then calls `start.sh` to run the main service, all orchestrated by the central API communicating with the Streamlit UI and MongoDB.

Is there any specific component or process you would like further details on?