from fastapi import FastAPI
from pydantic import BaseModel

from emojify.utils.get_emoji import get_emoji

app = FastAPI()


class Body(BaseModel):
    sentence: str


@app.post("/emoji")
async def get_emoji_endpoint(body: Body):
    emoji_output = get_emoji(body.sentence)
    return {"emoji": emoji_output}
