from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

# Create static directory for saved images
os.makedirs("web_logs", exist_ok=True)

# Store reports in memory (up to 10)
reports = []

@app.post("/api/report")
async def report(timestamp: str = Form(...), description: str = Form(...), image: UploadFile = Form(...)):
    img_filename = f"{timestamp}.jpg"
    img_path = os.path.join("web_logs", img_filename)

    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    reports.append({
        "timestamp": timestamp,
        "description": description,
        "image_path": img_path
    })

    # Limit to last 10
    if len(reports) > 10:
        reports.pop(0)

    return {"status": "received"}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Police Surveillance Dashboard</title>
        <style>
            body {
                background-color: #0e1117;
                font-family: 'Segoe UI', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
            h1 {
                color: #00bfff;
                text-align: center;
                font-size: 40px;
                text-shadow: 1px 1px #000;
            }
            .card {
                background-color: #1c1f26;
                border-left: 6px solid #00bfff;
                border-radius: 10px;
                margin-bottom: 20px;
                padding: 15px;
                box-shadow: 0px 0px 10px #007acc;
            }
            .card img {
                width: 100%;
                max-width: 400px;
                border-radius: 6px;
                margin-top: 10px;
            }
            .timestamp {
                font-size: 14px;
                color: #aaa;
            }
            .description {
                font-size: 18px;
                font-weight: bold;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1> POLICE INCIDENT LOG </h1>
    """

    # Display only the 10 latest reports
    latest_reports = reports[-10:]

    for report in reversed(latest_reports):
        img_path = f"/web_logs/{os.path.basename(report['image_path'])}"
        html += f"""
        <div class="card">
            <div class="timestamp">{report['timestamp']}</div>
            <div class="description">{report['description']}</div>
            <img src="{img_path}" alt="suspect">
        </div>
        """

    html += """
    </body>
    </html>
    """
    return html

# Serve uploaded images
app.mount("/web_logs", StaticFiles(directory="web_logs"), name="web_logs")
