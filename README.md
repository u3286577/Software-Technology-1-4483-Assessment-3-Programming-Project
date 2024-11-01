# Production Application - Predictive model for gold ETF price

This project runs a Flask web application within a Docker container using `docker-compose`.

To pull this repo, use `git clone [link-to.repo]`

Its recommended to run this container natively on linux or through WSL for the best experience.

## Quick Start

1. **Install Prerequisites**: Docker and Docker Compose
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   ```

2. **Build and Run the Application**
   ```bash
   sudo docker-compose up --build
   ```

3. **Access the Application**:
   Open your browser and go to:
   ```
   http://localhost
   ```

4. **Stopping the Application**
   ```bash
   sudo docker-compose down
   ```
   (or ctrl+c if app still running in foreground)

