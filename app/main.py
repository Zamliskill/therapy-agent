
# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.therapy_agent import langgraph_app

app = FastAPI()

class UserMessage(BaseModel):
    user_id: str
    name: str = None
    message: str

@app.post("/chat")
def chat(data: UserMessage):
    state = {
        "user_id": data.user_id,
        "name": data.name,
        "message": data.message
    }
    result = langgraph_app.invoke(state)
    return JSONResponse(
        content={
            "name": result["name"],
            "emotion": result["emotion"],
            "message": result["response"]
        },
        media_type="application/json; charset=utf-8"
    )

