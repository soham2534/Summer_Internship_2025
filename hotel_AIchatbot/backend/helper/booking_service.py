import re
import json
from datetime import datetime
from fastapi import HTTPException
from helper.models import BookingDetails

class BookingService:
    def __init__(self, hotel_service, ollama_service, session_manager):
        self.hotel_service = hotel_service
        self.ollama_service = ollama_service
        self.session_manager = session_manager
        self.date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\b\d{4}-\d{2}-\d{2}',
            r'\bfrom\s+.*?\s+to\s+',
            r'\bcheck.?in.*?\d',
            r'\bcheck.?out.*?\d'
        ]
    
    async def process_chat_message(self, session_id: str, message: str):
        """Process a chat message and return response"""
        session = self.session_manager.get_session_data(session_id)
        self.session_manager.add_message(session_id, "user", message)
        
        # Check for location and dates in initial step
        if session["current_step"] == "initial":
            return await self._handle_initial_step(session_id, message, session)
        else:
            return await self._handle_booking_steps(session_id, message, session)
    
    async def _handle_initial_step(self, session_id: str, message: str, session):
        """Handle the initial step of the conversation"""
        location = self.hotel_service.extract_location_from_message(message)
        has_dates = any(re.search(pattern, message.lower()) for pattern in self.date_patterns)
        
        if location and has_dates:
            session["user_location"] = location
            session["user_dates"] = True
            session["current_step"] = "showing_hotels"
            self.session_manager.update_session_data(session_id, "user_location", location)
            self.session_manager.update_session_data(session_id, "user_dates", True)
            self.session_manager.update_session_data(session_id, "current_step", "showing_hotels")
            
            reply = self.hotel_service.format_hotel_list(location)
            self.session_manager.add_message(session_id, "assistant", reply)
            
            filtered_hotels = self.hotel_service.get_hotels_by_location(location)
            
            return {
                "reply": reply,
                "hotels": filtered_hotels,
                "selected_hotel_details": None,
                "step": session["current_step"],
                "image_url": None,
            }
        else:
            
            try:
                conversation = self.session_manager.get_session(session_id)
                reply = self.ollama_service.chat(conversation)
                self.session_manager.add_message(session_id, "assistant", reply)
            except Exception as e:
                print(f"Ollama chat error: {e}")
                # Fallback responses
                if not location and not has_dates:
                    reply = "I'd be happy to help you find a hotel! Could you please tell me which location you're interested in and your travel dates?"
                elif not location:
                    reply = "Great! I see you have dates in mind. Which location would you like to stay in?"
                elif not has_dates:
                    reply = f"Perfect! I can help you find hotels in {location}. What are your check-in and check-out dates?"
                self.session_manager.add_message(session_id, "assistant", reply)
            
            return {
                "reply": reply,
                "hotels": None,
                "selected_hotel_details": None,
                "step": session["current_step"],
                "image_url": None,
            }
    
    async def _handle_booking_steps(self, session_id: str, message: str, session):
        """Handle booking steps after initial conversation"""
        try:
            conversation = self.session_manager.get_session(session_id)
            reply = self.ollama_service.chat(conversation)
            self.session_manager.add_message(session_id, "assistant", reply)
            
            # Check if a hotel has been selected
            if session["selected_hotel"] is None and session["current_step"] == "showing_hotels":
                reply = await self._handle_hotel_selection(session_id, message, session, reply)
            
            # Extract booking details
            if session["selected_hotel"]:
                reply = await self._extract_booking_details(session_id, message, session, reply)
            
        except Exception as e:
            print(f"Ollama chat error: {e}")
            reply = "I'm sorry, I'm having trouble processing your request. Please try again."
            self.session_manager.add_message(session_id, "assistant", reply)
        
        # Prepare response
        image_url = None
        if session.get("selected_hotel_image_url") and not session.get("image_sent"):
            image_url = session["selected_hotel_image_url"]
            self.session_manager.update_session_data(session_id, "image_sent", True)
        
        response_data = {
            "reply": reply,
            "hotels": None,
            "selected_hotel_details": session["selected_hotel_details"] if session["current_step"] != "initial" else None,
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
            return await self.confirm_booking(session_id, booking_details)
        
        return response_data
    
    async def _handle_hotel_selection(self, session_id: str, message: str, session, reply):
        """Handle hotel selection logic"""
        location_hotels = self.hotel_service.get_hotels_by_location(session["user_location"])
        
        for hotel in location_hotels:
            if hotel["hotel_name"].lower() in message.lower():
                self.session_manager.update_session_data(session_id, "selected_hotel", hotel["hotel_name"])
                self.session_manager.update_session_data(session_id, "selected_hotel_details", hotel)
                self.session_manager.update_session_data(session_id, "selected_hotel_image_url", hotel.get("image_url"))
                self.session_manager.update_session_data(session_id, "current_step", "check_in")
                
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
        
        return reply
    
    async def _extract_booking_details(self, session_id: str, message: str, session, reply):
        """Extract booking details from user messages"""
        if session["current_step"] == "check_in" and session["check_in"] is None:
            reply = self._handle_check_in_date(session_id, message, session)
        elif session["current_step"] == "check_out" and session["check_out"] is None:
            reply = self._handle_check_out_date(session_id, message, session)
        elif session["current_step"] == "num_guests" and session["num_guests"] is None:
            reply = self._handle_guest_count(session_id, message, session)
        elif session["current_step"] == "guest_names" and len(session["guest_names"]) < session["num_guests"]:
            reply = self._handle_guest_names(session_id, message, session)
        elif session["current_step"] == "phone" and session["phone"] is None:
            reply = self._handle_phone_number(session_id, message, session)
        
        return reply
    
    def _handle_check_in_date(self, session_id: str, message: str, session):
        """Handle check-in date extraction"""
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", message)
        if date_match:
            try:
                datetime.strptime(date_match.group(0), "%Y-%m-%d")
                self.session_manager.update_session_data(session_id, "check_in", date_match.group(0))
                self.session_manager.update_session_data(session_id, "current_step", "check_out")
                return f"Check-in date set to {date_match.group(0)}. Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
            except ValueError:
                return "Invalid date format. Please provide your check-in date in YYYY-MM-DD format (e.g., 2025-05-20)."
        else:
            return "Please provide your check-in date in YYYY-MM-DD format (e.g., 2025-05-20)."
    
    def _handle_check_out_date(self, session_id: str, message: str, session):
        """Handle check-out date extraction"""
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", message)
        if date_match:
            try:
                datetime.strptime(date_match.group(0), "%Y-%m-%d")
                self.session_manager.update_session_data(session_id, "check_out", date_match.group(0))
                self.session_manager.update_session_data(session_id, "current_step", "num_guests")
                return f"Check-out date set to {date_match.group(0)}. How many guests will be staying? (Max: {session['selected_hotel_details']['number_of_guests']})"
            except ValueError:
                return "Invalid date format. Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
        else:
            return "Please provide your check-out date in YYYY-MM-DD format (e.g., 2025-05-25)."
    
    def _handle_guest_count(self, session_id: str, message: str, session):
        """Handle guest count extraction"""
        num_match = re.search(r"\d+", message)
        if num_match:
            num_guests = int(num_match.group(0))
            max_guests = session["selected_hotel_details"]["number_of_guests"]
            if num_guests > max_guests:
                return f"Sorry, this hotel can only accommodate up to {max_guests} guests. Please provide a number of guests within this limit."
            elif num_guests <= 0:
                return "Please provide a valid number of guests (greater than 0)."
            else:
                self.session_manager.update_session_data(session_id, "num_guests", num_guests)
                self.session_manager.update_session_data(session_id, "current_step", "guest_names")
                return f"Got it, {num_guests} guests. Please provide the name of guest 1."
        else:
            return "Please provide the number of guests as a number (e.g., 2)."
    
    def _handle_guest_names(self, session_id: str, message: str, session):
        """Handle guest names collection"""
        name = message.strip()
        if name:
            session["guest_names"].append(name)
            self.session_manager.update_session_data(session_id, "guest_names", session["guest_names"])
            
            if len(session["guest_names"]) < session["num_guests"]:
                return f"Guest {len(session['guest_names'])} name recorded as {name}. Please provide the name of guest {len(session['guest_names']) + 1}."
            else:
                self.session_manager.update_session_data(session_id, "current_step", "phone")
                return f"Guest {len(session['guest_names'])} name recorded as {name}. All guest names collected. Please provide your phone number in XXX-XXX-XXXX format (e.g., 123-456-7890)."
        else:
            return f"Please provide the name of guest {len(session['guest_names']) + 1}."
    
    def _handle_phone_number(self, session_id: str, message: str, session):
        """Handle phone number extraction"""
        phone_match = re.search(r"\d{3}-\d{3}-\d{4}", message)
        if phone_match:
            self.session_manager.update_session_data(session_id, "phone", phone_match.group(0))
            self.session_manager.update_session_data(session_id, "current_step", "completed")
            return "Thank you for providing your details. I'll process your booking now."
        else:
            return "Please provide your phone number in XXX-XXX-XXXX format (e.g., 123-456-7890)."
    
    async def confirm_booking(self, session_id: str, details: BookingDetails):
        """Confirm a booking"""
        # Validate dates
        try:
            check_in_date = datetime.strptime(details.check_in, "%Y-%m-%d")
            check_out_date = datetime.strptime(details.check_out, "%Y-%m-%d")
            if check_out_date <= check_in_date:
                raise ValueError("Check-out date must be after check-in date")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        