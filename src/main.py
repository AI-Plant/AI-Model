from fastapi import FastAPI

app = FastAPI(docs_url="/docs", openapi_url="/open-api-docs")

@app.get("/")
async def getHello():
    return "Hello, World!"