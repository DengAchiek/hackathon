# GhostDemand AI

**Demand beyond the sale.** A retail and e-commerce assistant that turns unsuccessful customer requests into evidence for what to stock, change and follow up on next.

## Project Components

This repository contains the GhostDemand AI Flask application and a Next.js landing page.

### 1. Flask application — backend and dashboard

The Flask application provides the main GhostDemand AI workflow, including request capture, product and blocker extraction, an opportunity dashboard, follow-up management, CSV import/export, and machine-learning classification.

To run the Flask application, follow the **Run locally** instructions below. By default, it runs at `http://127.0.0.1:5000`.

### 2. Next.js landing page

The Next.js frontend is located in `ghost-demand-ai-site/`. It provides the GhostDemand AI landing page and login-form interface.

To run the Next.js project, open a terminal in the project folder:

```bash
cd ghost-demand-ai-site
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

**Note:** The login forms are currently frontend interfaces. Real authentication requires backend integration.

## Project Documentation

- [Hackathon overview](HACKATHON.md)
- [Machine-learning roadmap](ML_ROADMAP.md)
- [Step-by-step ML workflow](ml_workflow/README.md)
- [Business integration plan](BUSINESS_INTEGRATION.md)

## Run the Flask Application Locally

Python 3.10 or newer:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and start the application:

```bash
pip install -r requirements.txt
flask --app app train-model
python app.py
```

Open `http://127.0.0.1:5000` in your browser.
