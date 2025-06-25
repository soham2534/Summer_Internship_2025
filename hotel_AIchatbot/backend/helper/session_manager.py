class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_data = {}
        self.system_prompt = """
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
    
    def initialize_session(self, session_id: str):
        """Initialize a new session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = [{"role": "system", "content": self.system_prompt}]
            self.session_data[session_id] = {
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
    
    def get_session(self, session_id: str):
        """Get session conversation history"""
        self.initialize_session(session_id)
        return self.sessions[session_id]
    
    def get_session_data(self, session_id: str):
        """Get session data"""
        self.initialize_session(session_id)
        return self.session_data[session_id]
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session"""
        self.initialize_session(session_id)
        self.sessions[session_id].append({"role": role, "content": content})
    
    def update_session_data(self, session_id: str, key: str, value):
        """Update session data"""
        self.initialize_session(session_id)
        self.session_data[session_id][key] = value
    
    def reset_last_message(self, session_id: str):
        """Reset the last message in the session"""
        if session_id in self.sessions:
            conversation = self.sessions[session_id]
            if len(conversation) >= 2:
                conversation = conversation[:-2]
                self.sessions[session_id] = conversation
                return {"status": "Last user-bot message deleted successfully"}
            else:
                self.sessions[session_id] = []
                return {"status": "Conversation cleared (less than 2 messages)"}
        else:
            return {"status": "Session not found"}, 404