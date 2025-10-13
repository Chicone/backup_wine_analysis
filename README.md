# Wine Analysis Interface

This branch contains an interactive dashboard designed to run experiments on multiple GC-MS wine datasets.  
The dashboard provides all the functionality needed to execute the complete wine analysis workflow — including 
model training, evaluation, and visualization.  
It allows users to train different machine-learning models, compute classification accuracies, generate confusion 
matrices, and visualize results across all wine families considered in the project.

## 🧭 Getting Started

To use the dashboard locally, first clone the repository and switch to the correct branch.

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Chicone/wine_analysis.git
```

This will create a local folder named `wine_analysis` containing all project files.

### 2. Navigate into the project directory

```bash
cd wine_analysis
```

### 3. (Optional) Verify branch

If you’re not already on the web interface branch, switch to it with:

```bash
git checkout main
```

You now have a complete local copy of the project and can proceed to start the backend and frontend as described below.

---


## 📄 Documentation

The dashboard contains its own Docs page
Alternatively, the general documentation of the Wine Analysis Package can be found here: 
👉 [View the documentation](https://chicone.github.io/backup_wine_analysis/)


## 🚀 Running the Wine Analysis Dashboard

This project includes two main components:

- **Backend API** — built with [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend Dashboard** — built with [React](https://react.dev/)

Both must be running simultaneously for the dashboard to function.

---

### 🧩 1. Start the Backend (FastAPI)

Open a terminal and run:

```bash
cd ~/PycharmProjects/wine_analysis/api_web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on [http://localhost:8000](http://localhost:8000).

API documentation:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)


> ⚠️ Note: These documentation endpoints are provided by FastAPI and are only available on **port 8000**, not on the frontend (port 3000).

---

### 💻 2. Start the Frontend (React)

In another terminal window, run:

```bash
cd ~/PycharmProjects/wine_analysis/frontend
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


