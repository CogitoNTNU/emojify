from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from emojify.utils.get_emoji import get_emoji


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def serve():
    print("Starting server")


if __name__ == "__main__":
    serve()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class Body:
    sentence: str


@app.post("/")
async def submit_text(body: Body):
    emoji_output = get_emoji(body.sentence)
    return {"emoji": emoji_output}
    # return templates.TemplateResponse("index.html", {"request": request, "text": text, "message": emoji_output})


