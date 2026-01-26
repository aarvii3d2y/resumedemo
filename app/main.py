from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv
from .utils import extract_text_from_pdf, analyze_resume

load_dotenv()

app = FastAPI(title="AI Resume Screener")

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    # 1. Read File
    contents = await resume_file.read()
    
    # 2. Extract Text
    resume_text = ""
    if resume_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(contents)
    else:
        # Fallback for text/markdown files
        try:
            resume_text = contents.decode("utf-8")
        except:
             return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Could not read file. Please upload a valid PDF or text file."
            })
            
    if not resume_text.strip():
         return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Could not extract text from the resume."
        })

    # 3. Analyze
    result = analyze_resume(resume_text, job_description)
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "filename": resume_file.filename
    })

@app.get("/search_page", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    job_description: str = Form(...),
    csv_file: UploadFile = File(None)
):
    from .utils import search_resumes
    import pandas as pd
    from io import StringIO, BytesIO

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

    results = search_resumes(job_description, df=df)
    
    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": results,
        "job_description": job_description
    })
