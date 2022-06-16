from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Greetings!"}
