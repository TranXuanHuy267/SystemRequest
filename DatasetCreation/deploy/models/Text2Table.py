from typing import Dict, Union

from pydantic import BaseModel, Field


class Text2TableRequest(BaseModel):
    input: str = Field(
        examples=[
            "Mở biểu đồ",
            "Mở dashboard",
            "Mở doanh thu cộng ngang"
        ],
        description="Câu lệnh cho hệ thống vSDS"
    )

class Text2TableResponse(BaseModel):
    output: str
