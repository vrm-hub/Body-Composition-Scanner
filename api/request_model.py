from pydantic import BaseModel


class PredictRequest(BaseModel):
    height: int
    weight: int
    gender: str
    age: int
    file_front: bytes
    file_left: bytes
