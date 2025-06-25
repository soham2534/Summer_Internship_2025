import os
import json
import re
import uuid
from datetime import datetime
import requests
from fastapi import HTTPException
from gtts import gTTS
from pydantic import BaseModel
from .config import load_config

class BookingDetails(BaseModel):
    hotel_name: str
    room_type: str
    check_in: str
    check_out: str
    guests: int
    guest_names: list[str]
    phone: str
    location: str

# Load configuration
config = load_config()
OLLAMA_HOST = config["ollama_host"]
OLLAMA_MODEL = config["ollama_model"]
HOTEL_JSON_PATH = config["hotel_json_path"]
AUDIO_DIR = config["audio_dir"]
DATE_PATTERNS = config["date_pattern"]

# Load hotel data
try:
    if not os.path.exists(os.path.dirname(HOTEL_JSON_PATH)):
        os.makedirs(os.path.dirname(HOTEL_JSON_PATH))
    with open(HOTEL_JSON_PATH, 'r') as f:
        hotels = json.load(f)
    if not isinstance(hotels, list):
        raise ValueError("Hotel data must be a list of hotel objects")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load hotel data: {e}")

# System prompt
SYSTEM_PROMPT = """
You are a hotel booking assistant for a hotel. Follow this flow:
1. Welcome the user and ask for their hotel booking requirements (e.g., location and dates).
2. When the user mentions BOTH a location AND dates (e.g., "I need a hotel in Gujarat from May 20 to May 25"), respond with available hotel options. DO NOT show hotels unless BOTH location and dates are mentioned together.
3. Once the user selects a hotel by mentioning its name, confirm the selection and start collecting booking details one at a time:
   - First, ask for the check-in date (format: YYYY-MM-DD). If the user provides it, confirm and proceed.
   - Next, ask for the check-out date (format: YYYY-MM-DD). If provided, confirm and proceed.
   - Then, ask for the number of guests. If provided, confirm and proceed. Ensure the number of guests does not exceed the hotel's capacity.
   - Then, ask for the guest names (one name at a time, e.g., "John Doe", then "Jane Smith"). Collect names based on the number of guests.
   - Finally, ask for the phone number (format: XXX-XXX-XXXX). If provided, confirm and proceed.
4. After collecting each detail, confirm it and ask for the next one. If the user provides an invalid format, ask them to provide it again in the correct format.
5. Once all details are collected, give a summary of the booking details and ask for confirmation.

IMPORTANT: Only show hotel options when BOTH location AND dates are mentioned in the same message or conversation context.
"""

def ollama_chat(messages, temperature=0.7, max_tokens=700):
    """Interact with Ollama API for chat responses."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        if "message" not in result or "content" not in result["message"]:
            raise ValueError("Unexpected Ollama response format")
        return result["message"]["content"].strip()
    except requests.RequestException as e:
        print(f"Ollama chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama chat error: {e}")

def get_available_locations():
    """Extract all unique city and state names from hotel data."""
    locations_set = set()
    for hotel in hotels:
        location_parts = hotel["location"].lower().split(",")
        for part in location_parts:
            clean = part.strip()
            if clean:
                locations_set.add(clean)
    return list(locations_set)

def extract_location_from_message(message: str) -> str | None:
    """Scalable extraction of location by matching with hotel data locations."""
    message_lower = message.lower()
    for keyword in get_available_locations():
        if keyword in message_lower:
            return keyword.title()
    return None

def format_hotel_list(location=None):
    """Format a list of hotels based on location."""
    filtered_hotels = [
        hotel for hotel in hotels
        if not location or location.lower() in hotel['location'].lower()
    ]
    if not filtered_hotels:
        return f"Sorry, we don't have any hotels available in {location}. Please try a different location like Dwarka, Ahmedabad, Vadodara, Rajkot, or other major cities."
    formatted_list = "Fantastic! Here are some available hotel options that match your requirements:\n\n"
    for idx, hotel in enumerate(filtered_hotels, 1):
        formatted_list += f"{idx}. {hotel['hotel_name']} – ${hotel['price_per_night']}/night\n"
        formatted_list += f"    Location: {hotel['location']}\n"
        formatted_list += f"    Capacity: Up to {hotel['number_of_guests']} guests\n"
        description = hotel.get("description", "").strip()
        first_sentence = description.split(".")[0].strip() if description else "No description available"
        formatted_list += f"    {first_sentence}. "
        if "single" in description.lower():
            formatted_list += "This single room features:\n"
        elif "double" in description.lower() or "twin" in hotel["hotel_name"].lower():
            formatted_list += "This double room offers:\n"
        elif "suite" in hotel["hotel_name"].lower():
            formatted_list += "This suite includes:\n"
        else:
            formatted_list += "This room includes:\n"
        bullets = []
        if hotel.get("amenities"):
            bullets.extend(hotel["amenities"][:3])
        if len(bullets) < 3 and hotel.get("facilities"):
            bullets.extend(hotel["facilities"][:3 - len(bullets)])
        if not bullets:
            bullets = ["Modern furnishings", "Great location", "Exceptional service"]
        for bullet in bullets:
            formatted_list += f"    - {bullet}\n"
        formatted_list += "\n"
    formatted_list += "Which hotel would you like to choose?"
    return formatted_list

def generate_audio(reply: str, audio_dir: str) -> str:
    """Generate audio file from text using gTTS."""
    os.makedirs(audio_dir, exist_ok=True)
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(audio_dir, f"{audio_id}.mp3")
    try:
        tts = gTTS(text=reply, lang='en', slow=False)
        tts.save(audio_path)
        return f"/audio/{audio_id}.mp3"
    except Exception as e:
        print(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio response")

class Message(BaseModel):
    message: str

def handle_chat(session_id: str, msg: Message, sessions: dict, session_data: dict):
    """Handle the chat endpoint logic."""
    if not msg.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Initialize session
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

    # Handle conversation flow
    if session["current_step"] == "initial":
        location = extract_location_from_message(msg.message)
        has_dates = any(re.search(pattern, msg.message.lower()) for pattern in DATE_PATTERNS)
        if location and has_dates:
            session["user_location"] = location
            session["user_dates"] = True
            session["current_step"] = "showing_hotels"
            reply = format_hotel_list(location)
            sessions[session_id].append({"role": "assistant", "content": reply})
            audio_url = generate_audio(reply, AUDIO_DIR)
            filtered_hotels = [
                hotel for hotel in hotels
                if location.lower() in hotel['location'].lower()
            ]
            return {
                "reply": reply,
                "hotels": filtered_hotels,
                "selected_hotel_details": None,
                "audio_url": audio_url,
                "step": session["current_step"],
                "image_url": None,
            }
        else:
            try:
                reply = ollama_chat(sessions[session_id])
                sessions[session_id].append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f"Ollama chat error: {e}")
                if not location and not has_dates:
                    reply = "I'd be happy to help you find a hotel! Could you please tell me which location you're interested in and your travel dates?"
                elif not location:
                    reply = "Great! I see you have dates in mind. Which location would you like to stay in?"
                elif not has_dates:
                    reply = f"Perfect! I can help you find hotels in {location}. What are your check-in and check-out dates?"
                sessions[session_id].append({"role": "assistant", "content": reply})
    else:
        try:
            reply = ollama_chat(sessions[session_id])
            sessions[session_id].append({"role": "assistant", "content": reply})
            if session["selected_hotel"] is None and session["current_step"] == "showing_hotels":
                location_hotels = [
                    hotel for hotel in hotels
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

    audio_url = generate_audio(reply, AUDIO_DIR)
    image_url = None
    if session.get("selected_hotel_image_url") and not session.get("image_sent"):
        image_url = session["selected_hotel_image_url"]
        session["image_sent"] = True

    response_data = {
        "reply": reply,
        "hotels": None,
        "selected_hotel_details": session["selected_hotel_details"] if session["current_step"] != "initial" else None,
        "audio_url": audio_url,
        "step": session["current_step"],
        "image_url": image_url,
    }

    if session["current_step"] == "completed":
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
        return handle_confirm_booking(session_id, booking_details, sessions)

    return response_data

def handle_confirm_booking(session_id: str, details: BookingDetails, sessions: dict):
    """Handle the confirm booking endpoint logic."""
    try:
        check_in_date = datetime.strptime(details.check_in, "%Y-%m-%d")
        check_out_date = datetime.strptime(details.check_out, "%Y-%m-%d")
        if check_out_date <= check_in_date:
            raise ValueError("Check-out date must be after check-in date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if details.guests != len(details.guest_names):
        raise HTTPException(status_code=400, detail="Number of guests must match the number of guest names")

    selected_hotel = next((hotel for hotel in hotels if hotel['hotel_name'] == details.hotel_name), None)
    if not selected_hotel:
        raise HTTPException(status_code=404, detail="Selected hotel not found")
    if details.guests > selected_hotel["number_of_guests"]:
        raise HTTPException(status_code=400, detail=f"Number of guests ({details.guests}) exceeds hotel capacity ({selected_hotel['number_of_guests']})")

    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
        reply = ollama_chat(sessions[session_id], temperature=0.7, max_tokens=500)
        sessions[session_id].append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"Ollama chat error: {e}")
        reply = (
            f"Booking confirmed for {details.hotel_name}!\n"
            f"Location: {details.location}\n"
            f"Check-in: {details.check_in}, Check-out: {details.check_out}\n"
            f"Guests: {details.guests} ({', '.join(details.guest_names)})\n"
            f"Phone: {details.phone}"
        )

    json_data = None
    try:
        json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_data = json.loads(json_str)
            if not all(key in json_data for key in ["hotel_name", "check_in", "check_out", "guests", "guest_names", "phone", "location"]):
                raise ValueError("Incomplete booking JSON")
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"JSON extraction error: {e}")
        json_data = {
            "hotel_name": details.hotel_name,
            "check_in": details.check_in,
            "check_out": details.check_out,
            "guests": details.guests,
            "guest_names": details.guest_names,
            "phone": details.phone,
            "location": details.location
        }

    try:
        delta = check_out_date - check_in_date
        nights = delta.days
        price_per_night = selected_hotel['price_per_night']
        subtotal = nights * price_per_night
        tax = round(subtotal * 0.15, 2)
        total = subtotal + tax
        json_data["nights"] = nights
        json_data["price_per_night"] = price_per_night
        json_data["subtotal"] = subtotal
        json_data["tax"] = tax
        json_data["total"] = total
    except Exception as e:
        print(f"Price calculation error: {e}")

    audio_url = generate_audio(reply, AUDIO_DIR)
    return {
        "reply": reply,
        "final": True,
        "json": json_data,
        "audio_url": audio_url
    }