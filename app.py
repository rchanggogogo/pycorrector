# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/22 9:56 AM
==================================="""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from examples.corrector_pipeline import correct_text

fmt = '{time} - {name} - {level} - {message}'
logger.add('logs/app.log', format=fmt, level='DEBUG', rotation='500MB', compression='zip')

app = FastAPI()


@app.post("/corrector")
def corrector(sentence: str):
    corrected_text, detail = correct_text(sentence)
    return JSONResponse(content={'corrected_text': corrected_text, 'detail': detail})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='app:app', host='0.0.0.0', port=8085, log_level='debug', reload=True, workers=1)
