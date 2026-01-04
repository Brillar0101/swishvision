from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Creates API for the application with metadata
app = FastAPI(
    title="Swish Vision API",
    description="AI-powered basketball game analysis",
    version="0.1.0"
)

# Configures CORS allowing origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check at the root URL.
@app.get("/")
def root():
    return {"message": "Swish Vision API", "status": "running"}