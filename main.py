from fastapi import FastAPI
import uvicorn

from src.database import Base, engine, Session
from src.authenticate.router import auth_router


app = FastAPI()
app.include_router(auth_router)


@app.get('/healthchecker')
def healthchecker():
	return {"health status": "OK"}


def main() -> None:
	Base.metadata.create_all(engine)
	uvicorn.run(app, host='127.0.0.1', port=8000)


if __name__ == '__main__':
	main()