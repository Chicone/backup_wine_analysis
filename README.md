# 🍷 Wine Analysis Interface

This branch contains an interactive dashboard designed to run experiments on multiple GC-MS wine datasets.  
The dashboard provides all the functionality needed to execute the complete wine analysis workflow — including 
model training, evaluation, and visualization.  
It allows users to train different machine-learning models, compute classification accuracies, generate confusion 
matrices, and visualize results across all wine families considered in the project.

---

## 🧭 Getting Started

### 1. Clone the Repository

Open a terminal (or PowerShell on Windows) and run:

```bash
git clone https://github.com/pougetlab/wine_analysis.git
cd wine_analysis
git checkout main
```

This will create a local folder named `wine_analysis` containing the web interface branch.

---

### 2. Install Prerequisites

Make sure the following tools are installed on your system:

- **Python 3.9+**
- **pip** (Python package manager)
- **Node.js** and **npm** (for the frontend)

#### Check if they’re installed:
```bash
python --version        # or python3 --version
pip --version
node --version
npm --version
```

If any of these commands fail, install the missing tools as follows:

#### 🐍 Python and pip
- Download and install from [https://www.python.org/downloads/](https://www.python.org/downloads/)
- When installing on Windows, make sure to check **“Add Python to PATH.”**

#### ⚙️ Node.js and npm
- Download from [https://nodejs.org/](https://nodejs.org/).  
  npm (Node Package Manager) comes **bundled** with Node.js automatically.  
  Choose the **LTS (Long-Term Support)** version for stability.

Alternatively, on Linux you can install it from the terminal:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install nodejs npm

# Fedora
sudo dnf install nodejs npm

# macOS (with Homebrew)
brew install node
```

---

### 3. Set Up the Python Environment

It is strongly recommended to use a virtual environment for the backend to isolate dependencies.

#### On Linux or macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

Once the environment is active (you should see `(.venv)` in your terminal prompt),  
install the required Python dependencies — including **Uvicorn** to run the FastAPI server:

```bash
pip install -r requirements.txt
```

If you encounter missing dependencies (e.g., during model training or data visualization),  
you can install these additional commonly used packages:

```bash
pip install torch torchvision pynndescent netCDF4 seaborn umap-learn tqdm scikit-optimize pycairo fastapi uvicorn[standard] 
```

---

## 🚀 Running the Wine Analysis Dashboard

This project includes two main components:

- **Backend API** — built with [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend Dashboard** — built with [React](https://react.dev/)

Both must be running simultaneously for the dashboard to function.

---

### 🧩 1. Start the Backend (FastAPI)

Open a terminal and run:

```bash
cd api_web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on [http://localhost:8000](http://localhost:8000).

API documentation:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

> ⚠️ Note: These documentation endpoints are provided by FastAPI and are only available on **port 8000**, not on the frontend (port 3000).

---

### 💻 2. Start the Frontend (React)

In another terminal window (or Command Prompt on Windows), run:

```bash
cd frontend
npm install
npm start
```

The frontend will start on [http://localhost:3000](http://localhost:3000) and will automatically connect to the backend running on port 8000.

---

### 🔄 Summary

| Component | Command | URL |
|------------|----------|-----|
| **Backend (FastAPI)** | `uvicorn main:app --reload --host 0.0.0.0 --port 8000` | [http://localhost:8000](http://localhost:8000) |
| **Frontend (React)** | `npm start` | [http://localhost:3000](http://localhost:3000) |
| **Swagger UI (API Docs)** | — | [http://localhost:8000/docs](http://localhost:8000/docs) |

Press **Ctrl + C** in each terminal to stop the servers.

---

## 📄 Documentation

The dashboard includes its own documentation panel.  
For detailed explanations of the GC-MS pipelines, models, and visualization logic, refer to:  
👉 [https://pougetlab.github.io/wine_analysis/](https://pougetlab.github.io/wine_analysis/)
