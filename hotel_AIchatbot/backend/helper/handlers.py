from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
import uuid
import os
import json
import re
from gtts import gTTS
from datetime import datetime
from typing import Dict, Any
from .config import AUDIO_DIR, SYSTEM_PROMPT, DATE_PATTERNS
from .functions import ollama_chat, extract_location_from_message, format_hotel_list, HOTELS, AVAILABLE_LOCATION_KEYWORDS

# Store sessions for history, context, and conversational records
sessions: Dict[str, list] = {}
session_data: Dict[str, Dict[str, Any]] = {}
os.makedirs(AUDIO_DIR, exist_ok=True)

# Define the models for Messages and BookingDetails
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

async def handle_chat(session_id: str, msg: Message, confirm_booking_callback) -> Dict[str, Any]:
    if not msg.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        session_data[session_id] = {
            "selected_hotel": None,
            "selected_hotel_details": None,
            "check_in": None,
            "check_out": None,
            "num_guests": None,
            "guest_names": [],
            "phone": None,
            "current_step": "initial",
            "user_location": None,
            "user_dates": None,
            "image_sent": False
        }

    # Add user message to session
    sessions[session_id].append({"role": "user", "content": msg.message})
    session = session_data[session_id]

    # Check for location and dates
    if session["current_step"] == "initial":
        # Extract location
        location = extract_location_from_message(msg.message, AVAILABLE_LOCATION_KEYWORDS)
        # Check for dates
        has_dates = any(re.search(pattern, msg.message.lower()) for pattern in DATE_PATTERNS)
        # Only show hotels if location and dates are mentioned
        if location and has_dates:
            session["user_location"] = location
            session["user_dates"] = True
            session["current_step"] = "showing_hotels"
            reply = format_hotel_list(location)
            sessions[session_id].append({"role": "assistant", "content": reply})
            # Generate audio
            audio_id = str(uuid.uuid4())
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.mp3")
            try:
                tts = gTTS(text=reply, lang='en', slow=False)
                tts.save(audio_path)
            except Exception as e:
                print(f"TTS generation error: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate audio response")
            # Filter hotels by location for response
            filtered_hotels = [
                hotel for hotel in HOTELS
                if location.lower() in hotel['location'].lower()
            ]
            return {
                "reply": reply,
                "hotels": filtered_hotels,
                "selected_hotel_details": None,
                "audio_url": f"/audio/{audio_id}.mp3",
                "step": session["current_step"],
                "image_url": None,
            }
        else:
            # Use Ollama for natural conversation when not showing hotels
            try:
                reply = ollama_chat(sessions[session_id])
                sessions[session_id].append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f"Ollama chat error: {e}")
                # Fallback responses
                if not location and not has_dates:
                    reply = "I'd be happy to help you find a hotel! Could you please tell me which location you're interested in and your travel dates?"
                elif not location:
                    reply = "Great! I see you have dates in mind. Which location would you like to stay in?"
                elif not has_dates:
                    reply = f"Perfect! I can help you find hotels in {location}. What are your check-in and check-out dates?"
                sessions[session_id].append({"role": "assistant", "content": reply})
    else:
        try:
            # Get response from Ollama
            reply = ollama_chat(sessions[session_id])
            sessions[session_id].append({"role": "assistant", "content": reply})
            # Check if a hotel has been selected
            if session["selected_hotel"] is None and session["current_step"] == "showing_hotels":
                location_hotels = [
                    hotel for hotel in HOTELS
                    if session["user_location"].lower() in hotel['location'].lower()
                ]
                for hotel in location_hotels:
                    if hotel["hotel_name"].lower() in msg.message.lower():
                        session["selected_hotel"] = hotel["hotel_name"]
                        session["selected_hotel_details"] = hotel
                        session["selected_hotel_image_url"] = hotel.get("image_url")
                        session["current_step"] = "check_in"
                        amenities = ", ".join(hotel.get("amenities", []) or ["None"])
                        facilities = ", ".join(hotel.get("facilities", []) or ["None"])
                        reply = (
                            f"You selected {hotel['hotel_name']} in {hotel['location']}.\n"
                            f"Capacity: Up to {hotel['number_of_guests']} guests\n"
                            f"Amenities: {amenities}\n"
                            f"Facilities: {facilities}\n"
                            f"Please provide your check-in date in YYYY-MM-DD format (e.g., 2025-05-20)."
                        )
                        break
            # Extract booking details
            if session["selected_hotel"]:
                if session["current_step"] == "check_in" and session["check_in"] is None:
                    date_match = re.search(r"\d{4}-\d{2}-\d{2}", msg.message)
                    if date_match:
                        try:
                            datetime.strptime(date_match.group(0), "%Y-%m-%d")
                            session["check_in"] = date_match.group(0)
                            session["current_step"] = "check_out"
                            reply = f"Check-in date set to {session['check_in']}. Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
                        except ValueError:
                            reply = "Invalid date format. Please provide your check-in date in YYYY-MM-DD format (e.g., 2025-05-20)."
                    else:
                        reply = "Please provide your check-in date in YYYY-MM-DD format (e.g., 2025-05-20)."
                elif session["current_step"] == "check_out" and session["check_out"] is None:
                    date_match = re.search(r"\d{4}-\d{2}-\d{2}", msg.message)
                    if date_match:
                        try:
                            datetime.strptime(date_match.group(0), "%Y-%m-%d")
                            session["check_out"] = date_match.group(0)
                            session["current_step"] = "num_guests"
                            reply = f"Check-out date set to {session['check_out']}. How many guests will be staying? (Max: {session['selected_hotel_details']['number_of_guests']})"
                        except ValueError:
                            reply = "Invalid date format. Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
                    else:
                        reply = "Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
                elif session["current_step"] == "num_guests" and session["num_guests"] is None:
                    num_match = re.search(r"\d+", msg.message)
                    if num_match:
                        num_guests = int(num_match.group(0))
                        max_guests = session["selected_hotel_details"]["number_of_guests"]
                        if num_guests > max_guests:
                            reply = f"Sorry, this hotel can only accommodate up to {max_guests} guests. Please provide a number of guests within this limit."
                        elif num_guests <= 0:
                            reply = "Please provide a valid number of guests (greater than 0)."
                        else:
                            session["num_guests"] = num_guests
                            session["current_step"] = "guest_names"
                            reply = f"Got it, {session['num_guests']} guests. Please provide the name of guest 1."
                    else:
                        reply = "Please provide the number of guests as a number (e.g., 2)."
                elif session["current_step"] == "guest_names" and len(session["guest_names"]) < session["num_guests"]:
                    name = msg.message.strip()
                    if name:
                        session["guest_names"].append(name)
                        if len(session["guest_names"]) < session["num_guests"]:
                            reply = f"Guest {len(session['guest_names'])} name recorded as {name}. Please provide the name of guest {len(session['guest_names']) + 1}."
                        else:
                            session["current_step"] = "phone"
                            reply = f"Guest {len(session['guest_names'])} name recorded as {name}. All guest names collected. Please provide your phone number in XXX-XXX-XXXX format (e.g., 123-456-7890)."
                    else:
                        reply = f"Please provide the name of guest {len(session['guest_names']) + 1}."
                elif session["current_step"] == "phone" and session["phone"] is None:
                    phone_match = re.search(r"\d{3}-\d{3}-\d{4}", msg.message)
                    if phone_match:
                        session["phone"] = phone_match.group(0)
                        session["current_step"] = "completed"
                        reply = "Thank you for providing your details. I'll process your booking now."
                    else:
                        reply = "Please provide your phone number in XXX-XXX-XXXX format (e.g., 123-456-7890)."
        except Exception as e:
            print(f"Ollama chat error: {e}")
            reply = "I'm sorry, I'm having trouble processing your request. Please try again."
            sessions[session_id].append({"role": "assistant", "content": reply})

    # Generate audio
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.mp3")
    try:
        tts = gTTS(text=reply, lang='en', slow=False)
        tts.save(audio_path)
    except Exception as e:
        print(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio response")

    # Prepare response
    image_url = None
    if session.get("selected_hotel_image_url") and not session.get("image_sent"):
        image_url = session["selected_hotel_image_url"]
        session["image_sent"] = True

    response_data = {
        "reply": reply,
        "hotels": None,
        "selected_hotel_details": session["selected_hotel_details"] if session["current_step"] != "initial" else None,
        "audio_url": f"/audio/{audio_id}.mp3",
        "step": session["current_step"],
        "image_url": image_url,
    }

    if session["current_step"] == "completed":
        # Prepare booking details for confirmation
        booking_details = BookingDetails(
            hotel_name=session["selected_hotel"],
            room_type=session["selected_hotel_details"].get("room_type", session["selected_hotel_details"]["hotel_name"]),
            check_in=session["check_in"],
            check_out=session["check_out"],
            guests=session["num_guests"],
            guest_names=session["guest_names"],
            phone=session["phone"],
            location=session["selected_hotel_details"]["location"]
        )
        # Call the confirm endpoint internally
        confirmation_response = await confirm_booking_callback(session_id, booking_details)
        return confirmation_response

    return response_data

async def handle_confirm_booking(session_id: str, details: BookingDetails) -> Dict[str, Any]:
    # Validate dates
    try:
        check_in_date = datetime.strptime(details.check_in, "%Y-%m-%d")
        check_out_date = datetime.strptime(details.check_out, "%Y-%m-%d")
        if check_out_date <= check_in_date:
            raise ValueError("Check-out date must be after check-in date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate number of guests matches guest names
    if details.guests != len(details.guest_names):
        raise HTTPException(status_code=400, detail="Number of guests must match the number of guest names")

    # Validate hotel selection and guest capacity
    selected_hotel = next((hotel for hotel in HOTELS if hotel['hotel_name'] == details.hotel_name), None)
    if not selected_hotel:
        raise HTTPException(status_code=404, detail="Selected hotel not found")
    if details.guests > selected_hotel["number_of_guests"]:
        raise HTTPException(status_code=400, detail=f"Number of guests ({details.guests}) exceeds hotel capacity ({selected_hotel['number_of_guests']})")

    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Prepare booking message for the LLM
    booking_message = (
        f"User has selected {details.hotel_name} for booking.\n"
        f"Location: {details.location}\n"
        f"Check-in: {details.check_in}\n"
        f"Check-out: {details.check_out}\n"
        f"Number of guests: {details.guests}\n"
        f"Guest names: {', '.join(details.guest_names)}\n"
        f"Phone: {details.phone}\n"
        f"Please confirm the booking and return the details in the specified JSON format."
    )
    sessions[session_id].append({"role": "user", "content": booking_message})

    try:
        # Get response from Ollama
        reply = ollama_chat(sessions[session_id], temperature=0.7, max_tokens=500)
        sessions[session_id].append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"Ollama chat error: {e}")
        # Create fallback confirmation message
        reply = (
            f"Booking confirmed for {details.hotel_name}!\n"
            f"Location: {details.location}\n"
            f"Check-in: {details.check_in}, Check-out: {details.check_out}\n"
            f"Guests: {details.guests} ({', '.join(details.guest_names)})\n"
            f"Phone: {details.phone}"
        )

    # Extract or create JSON data
    json_data = None
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_data = json.loads(json_str)
            # Validate required fields
            if not all(key in json_data for key in ["hotel_name", "check_in", "check_out", "guests", "guest_names", "phone", "location"]):
                raise ValueError("Incomplete booking JSON")
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"JSON extraction error: {e}")
        # Create fallback JSON data
        json_data = {
            "hotel_name": details.hotel_name,
            "check_in": details.check_in,
            "check_out": details.check_out,
            "guests": details.guests,
            "guest_names": details.guest_names,
            "phone": details.phone,
            "location": details.location
        }

    # Calculate staying duration and total price
    try:
        delta = check_out_date - check_in_date
        nights = delta.days
        price_per_night = selected_hotel['price_per_night']
        subtotal = nights * price_per_night
        tax = round(subtotal * 0.15, 2)  # 15% tax
        total = subtotal + tax
        json_data["nights"] = nights
        json_data["price_per_night"] = price_per_night
        json_data["subtotal"] = subtotal
        json_data["tax"] = tax
        json_data["total"] = total
    except Exception as e:
        print(f"Price calculation error: {e}")

    # Generate audio
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}.mp3")
    try:
        tts = gTTS(text=reply, lang='en', slow=False)
        tts.save(audio_path)
    except Exception as e:
        print(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio response")

    return {
        "reply": reply,
        "final": True,
        "json": json_data,
        "audio_url": f"/audio/{audio_id}.mp3"
    }

async def handle_reset_last_message(session_id: str) -> Dict[str, Any]:
    if session_id in sessions:
        conversation = sessions[session_id]
        if len(conversation) >= 2:
            conversation = conversation[:-2]
            sessions[session_id] = conversation
            return {"status": "Last user-bot message deleted successfully"}
        else:
            sessions[session_id] = []
            return {"status": "Conversation cleared (less than 2 messages)"}
    else:
        return {"status": "Session not found"}, 404

async def handle_get_audio(filename: str) -> FileResponse:
    audio_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path)

async def handle_get_hotels() -> Dict[str, Any]:
    return {"hotels": HOTELS}