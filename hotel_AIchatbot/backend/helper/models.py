from pydantic import BaseModel, field_validator
from datetime import datetime
import re

class Message(BaseModel):
    message: str

class BookingDetails(BaseModel):
    hotel_name: str
    room_type: str
    check_in: str
    check_out: str
    guests: int
    guest_names: list[str]
    phone: str
    location: str

    @field_validator("check_in", "check_out")
    @classmethod
    def validate_dates(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format! Date format should be YYYY-MM-DD")
        return value

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        if not re.match(r"^\d{3}-\d{3}-\d{4}$", value):
            raise ValueError("Phone number must be in the XXX-XXX-XXXX format")
        return value