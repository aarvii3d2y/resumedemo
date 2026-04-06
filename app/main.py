from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="AI Resume Screener")

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/search_page", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    job_description: str = Form(...),
    csv_file: UploadFile = File(None)
):
    from .utils import get_top_candidates
    import pandas as pd
    from io import StringIO

    df = None
    if csv_file and csv_file.filename:
        content = await csv_file.read()
        try:
            # Try decoding as utf-8
            s = str(content, 'utf-8')
            data = StringIO(s)
            df = pd.read_csv(data)
        except Exception:
             # Fallback
             pass

    results = get_top_candidates(job_description,top_n = 10, df=df)
    
    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": results,
        "job_description": job_description
    })
