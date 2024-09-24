from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from .model_script import recommender_model
import asyncio
from ..llm_api.openai_script import (
    is_enough,
    is_gibberish,
    get_explanation,
    get_specifications,
)

ml_router = APIRouter()


class DataObject(BaseModel):
    experience: str
    education: str
    skills: str


@ml_router.get("/model")
async def mlEndpointTest():
    return "ml model endpoint test "


async def model_even_stream(text):

    await asyncio.sleep(1)
    yield "[CHECK]Checking if text is gibberish...\n"
    if await is_gibberish(text):
        yield "[ERROR]Failed: The information is gibberish.\n"
        raise HTTPException(
            status_code=400,
            detail="The information you have provided is simply gibberish.",
        )

    yield "[SUCCESS]Passed gibberish check. \n"
    await asyncio.sleep(1)  # Simulate some delay
    yield "[CHECK]Checking if data is enough...\n"
    if not await is_enough(text):
        yield "[ERROR]Failed: The information is not enough.\n"
        raise HTTPException(
            status_code=400, detail="The information you have provided is not enough."
        )
    yield "[SUCCESS]Passed 'enough data' check. \n"
    await asyncio.sleep(1)  # Simulate some delay
    yield "[CHECK]Running Recommendation model...\n"
    result = await recommender_model(text)
    yield "[SUCCESS]recommendation done. \n"
    await asyncio.sleep(1)
    yield "[TITLE]Recommended Occupations  \n"
    await asyncio.sleep(1)

    yield f"[CARD2]{result[0]}"
    await asyncio.sleep(1)
    yield f"[CARD2]{result[1]}"
    await asyncio.sleep(1)
    yield f"[CARD2]{result[2]}"
    await asyncio.sleep(1)
    # yield f"[START][DATA]"
    # await asyncio.sleep(1)
    # print(result)
    # for element in result:
    #     print(f"[DATA]{element}")
    #     await asyncio.sleep(1)
    # yield "[FINISHED]"
    await asyncio.sleep(1)
    yield "[CHECK]Generating explanation. \n"
    await asyncio.sleep(1)
    explanation = await get_explanation(text, result[0])
    yield f"[CARD]Explanation for {result[0]} : {explanation}\n"
    await asyncio.sleep(1)
    yield f"[CHECK]Generating career specialities."
    specification = await get_specifications(text, result[0])
    yield f"[CARD]Specifications of {result[0]}"
    await asyncio.sleep(1)
    yield f"[CARD]there is multiple options , {specification}\n"
    await asyncio.sleep(1)
    yield "[FINISH]Data proccessing is done"

    # return {"explanation": explanation, "specification": specification}


@ml_router.post("/model")
async def ml_data_object(object: DataObject):
    text = f"Experience : {object.experience} . Education : {object.education} , Skills : {object.skills}"
    return StreamingResponse(model_even_stream(text), media_type="text/plain")
