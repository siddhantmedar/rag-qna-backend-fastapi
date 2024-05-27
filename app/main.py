from fastapi import FastAPI,Body,Request,Response,Depends,status,HTTPException
from inference import fetch_answer 
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

import redis.asyncio as redis
from contextlib import asynccontextmanager

from math import ceil

REDIS_URL = "redis://localhost:6380"

async def custom_callback(request: Request, response: Response, pexpire: int):
    expire = ceil(pexpire / 1000)

    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        f"Too Many Requests. Retry after {expire} seconds.",
        headers={"Retry-After": str(expire)},
    )

@asynccontextmanager
async def lifespan(_: FastAPI):
    redis_connection = redis.from_url(REDIS_URL, encoding="utf8")
    await FastAPILimiter.init(
        redis=redis_connection,
        http_callback=custom_callback,
    )
    yield
    await FastAPILimiter.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.middleware("http")
async def my_middleware(request:Request, call_next):
    print("Entered inside middleware")
    
    response = await call_next(request)
    
    print("Returning from middleware")
    
    return response

@app.get("/")
def root():
    return {"message": "Fast API is running"}

@app.post("/",dependencies=[Depends(RateLimiter(times=2, seconds=10))])
def get_query_answer(query: str = Body(..., embed=True)):
    answer = fetch_answer(query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True)