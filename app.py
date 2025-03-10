from Hate_Speech.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from Hate_Speech.pipeline.prediction_pipeline import PredictionPipeline
from Hate_Speech.exception import CustomException
from Hate_Speech.contant import *

text:str = "what is deep learning and NLP?"

app = FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/trian")
async def training():
    try: 
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training Successful! ")
    except Exception as e:
        return Response(f"Error occured! {e}")
    
@app.get("/predict")
async def predict_route(text):
   try:
      obj = PredictionPipeline()
      text = obj.run_pipeline(text)
      return text
   except Exception as e:
      raise CustomException(e,sys) from e
   

if __name__ == '__main__':
    uvicorn.run(app,host=APP_HOST,port=APP_PORT)