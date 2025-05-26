import os

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from model_ml import (
    process_csv_file,
    process_single_record,
    save_corrected_answer,
    process_video_file,
    save_correct_video_answer
)


class TextInput(BaseModel):
    text: str


class CorrectedAnswer(BaseModel):
    text: str
    corrected_label: bool

class CorrectedVideoAnswer(BaseModel):
    prediction_result: str
    video_path: str


app = FastAPI()

templates = Jinja2Templates(directory="templates")

###
# "/truscanai" in route used for compatibles with Prod
###

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process-csv/")
@app.post("/truscanai/process-csv/")
async def upload_csv(file: UploadFile = File(...)):
    processed_file_path = process_csv_file(file)

    return FileResponse(
        processed_file_path, filename=os.path.basename(processed_file_path)
    )


@app.post("/process-text/")
@app.post("/truscanai/process-text/")
async def upload_text(text: TextInput):
    result = process_single_record(text.text)
    return {"prediction_percent": result}


@app.post("/correct-answer/")
@app.post("/truscanai/correct-answer/")
async def upload_text(ca: CorrectedAnswer):
    save_corrected_answer(ca.text, ca.corrected_label)
    return status.HTTP_200_OK

@app.post("/process-video/")
@app.post("/truscanai/process-video/")
async def upload_video(file: UploadFile = File(...)):
    result = await process_video_file(file)

    return {"prediction_result": result.result, "video_path": result.video_path}

@app.post("/correct-video-answer/")
@app.post("/truscanai/correct-video-answer/")
async def upload_text(ca: CorrectedVideoAnswer):
    save_correct_video_answer(ca.video_path, ca.prediction_result)
    return status.HTTP_200_OK

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
