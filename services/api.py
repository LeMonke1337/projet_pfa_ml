from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.routes import model_service


app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(model_service.ml_router, prefix="")


@app.get("/")
async def test_endpoint():
    return "message"
