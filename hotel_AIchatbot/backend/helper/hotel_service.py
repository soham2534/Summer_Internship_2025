import json
import os
from fastapi import HTTPException

class HotelService:
    def __init__(self, config):
        self.hotel_json_path = config["hotel_json_path"]
        self.hotels = self._load_hotels()
        self.available_location_keywords = self._get_available_locations()
    
    def _load_hotels(self):
        """Load hotel data from JSON file"""
        try:
            if not os.path.exists(os.path.dirname(self.hotel_json_path)):
                os.makedirs(os.path.dirname(self.hotel_json_path))
            
            with open(self.hotel_json_path, 'r') as f:
                hotels = json.load(f)
            
            if not isinstance(hotels, list):
                raise ValueError("Hotel data must be a list of hotel objects")
            
            return hotels
            
        except Exception as e:
            print(f"Error loading hotel data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load hotel data: {e}")
    
    def _get_available_locations(self):
        """Extract all unique city and state names from hotel data"""
        locations_set = set()
        for hotel in self.hotels:
            location_parts = hotel["location"].lower().split(",")
            for part in location_parts:
                clean = part.strip()
                if clean:
                    locations_set.add(clean)
        return list(locations_set)
    
    def extract_location_from_message(self, message: str) -> str | None:
        """Scalable extraction of location by matching with hotel data locations"""
        message_lower = message.lower()
        for keyword in self.available_location_keywords:
            if keyword in message_lower:
                return keyword.title()
        return None
    
    def get_hotels_by_location(self, location: str = None):
        """Get hotels filtered by location"""
        if not location:
            return self.hotels
        
        filtered_hotels = [
            hotel for hotel in self.hotels
            if location.lower() in hotel['location'].lower()
        ]
        return filtered_hotels
    
    def get_hotel_by_name(self, hotel_name: str):
        """Get a specific hotel by name"""
        return next((hotel for hotel in self.hotels if hotel['hotel_name'] == hotel_name), None)
    
    def get_all_hotels(self):
        """Get all hotels"""
        return self.hotels
    
    def format_hotel_list(self, location=None):
        """Format hotel list for display"""
        filtered_hotels = self.get_hotels_by_location(location)
        
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
            
            # Determine room type
            if "single" in description.lower():
                formatted_list += "This single room features:\n"
            elif "double" in description.lower() or "twin" in hotel["hotel_name"].lower():
                formatted_list += "This double room offers:\n"
            elif "suite" in hotel["hotel_name"].lower():
                formatted_list += "This suite includes:\n"
            else:
                formatted_list += "This room includes:\n"
            
            # Add bullet points
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