import sys
from pathlib import Path
import uvicorn

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    print("=" * 65)
    print(" SIGNBRIDGE: AI Indian Sign Language Recognition & Voice System")
    print(" Serving on http://127.0.0.1:8000")
    print(" Swagger Docs available at http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
