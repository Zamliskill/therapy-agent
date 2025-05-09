import os
import logging
import random
from datetime import datetime
from typing import TypedDict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph

# ---- LOGGING ---- #
logging.basicConfig(level=logging.INFO)

# ---- STATE ---- #
class TherapyState(TypedDict, total=False):
    user_id: str
    name: Optional[str]
    message: str
    emotion: Optional[str]
    dua: Optional[str]
    response: Optional[str]

# ---- ENV ---- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is missing in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# ---- MEMORY ---- #
memory = {}

# ---- FLAVOR ---- #
def generate_prompt_flavor():
    moods = ["gentle", "hopeful", "tender", "comforting", "reassuring", "sincere", "soft-spoken"]
    metaphors = [
        "like sunrise breaking through clouds",
        "like rain falling gently on dry land",
        "like a friend sitting silently beside you",
        "like warm hands around a cold heart",
        "like whispers of hope in a storm"
    ]
    emotion_frame = [
        "Speak as someone who has felt this pain too.",
        "Talk as if you're wrapping the person in a warm blanket of peace.",
        "Speak to their heart as a soul who cares deeply.",
        "Use words that feel like a calm sea after waves of distress."
    ]
    return f"{random.choice(moods).capitalize()} tone, {random.choice(metaphors)}, {random.choice(emotion_frame)}"

# ---- NODES ---- #
def set_user_memory(state: TherapyState) -> TherapyState:
    uid = state["user_id"]
    memory.setdefault(uid, {})
    if state.get("name"):
        memory[uid]["name"] = state["name"]
    state["name"] = memory[uid].get("name", "Friend")
    return state

def classify_emotion(state: TherapyState) -> TherapyState:
    user_msg = state["message"]
    prompt = f'''
User message: "{user_msg}"

Your task is to identify the user's **current emotional state**.
Respond in **just one word**, like: "horny", "ashamed", "overwhelmed", "peaceful", "hopeful", "sad", "happy","anxious"etc.
If unclear, respond with: "uncertain"
'''
    try:
        response = model.generate_content(prompt)
        emotion = response.text.strip().lower()
        if not emotion or len(emotion.split()) > 3:
            raise ValueError("Invalid emotion response")
        state["emotion"] = emotion if emotion != "uncertain" else None
    except Exception as e:
        logging.error(f"Emotion detection failed: {e}")
        state["emotion"] = None

    logging.info(f"Detected emotion: {state['emotion']}")
    return state

def route_based_on_emotion(state: TherapyState) -> str:
    return "emotional" if state.get("emotion") else "casual"

def fetch_dua(state: TherapyState) -> TherapyState:
    emotion = state["emotion"]
    prompt = f'''
Give a short authentic dua with Arabic diacritics (harakaat) and translation.
For the emotion: {emotion}

Rules:
- Use only authentic duas from Quran, Hadith, Seerah.
- Search for short duas.
- If no specific dua exists for this emotion, use a general comforting one but that should be authentic too.

Format:
Arabic: ...
Translation: ...
'''
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "Arabic:" not in text or "Translation:" not in text:
            raise ValueError("Incomplete format")
        state["dua"] = text
    except Exception as e:
        logging.error(f"Dua generation failed: {e}")
        fallback = "Arabic: اللَّهُمَّ آتِ نَفْسِي تَقْوَاهَا، وَزَكِّهَا أَنْتَ خَيْرُ مَنْ زَكَّاهَا، أَنْتَ وَلِيُّهَا وَمَوْلَاهَا\nTranslation: O Allah, grant my soul its piety and purify it. You are the best to purify it. You are its Guardian and Protector."
        state["dua"] = fallback
    return state

def generate_counseling(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    emotion = state.get("emotion", "neutral")
    user_msg = state["message"]
    tone_instruction = generate_prompt_flavor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dua_line = f"\nHere’s a short dua for you to softly recite:\n{state['dua']}" if state.get("dua") else ""

    prompt = f"""
    Detect user message language: if English, respond in English; if Roman Urdu, use Roman Urdu.
    You are Mustafa, an Islamic therapist. Write a warm, persuasive, and structured reply like a real therapist.
    To improve readability:
        - Break your message into short paragraphs.
        - Use bullet points (`-`) or numbered steps (1. 2. 3.) for clarity where helpful.
        - Use all CAPS to show emphasis emphasis. Avoid asterisks.
        - In Roman Urdu, use UPPERCASE for emphasis instead of bold.
        - Emphasize important lines using all caps.
        - Avoid long blocks of text—keep each section focused and skimmable.
        - Don't say I am Mustafa or your intro in response until the user asks about you.

    Things to Remember:
        - Don't use complex and difficult english words, use simple and easy words.
        - Use a warm, gentle, and soft tone.
        - Blend Seerah of Prophets, Hadith, Ayah, islamic history naturally (no references). Use soft, healing, human tone.
        - The response should make the user feel better and more connected to Allah.
        - Don't recommend any haram or unislamic things, and for haram things like haram relationships, music etc, tell the azaab for it and its consequences.
        - Avoid robotic responses.
        - The response should be like a real therapist, not a chatbot that make user calm and relaxed.
        - Include this dua in your reply if it exists with proper arabic script, diatrics (harkaat) by saying like here is dua for you or recite this dua, say according to struction and condition.

    User: {name}
    Emotion: {emotion}
    Message: \"{user_msg}\"
    Tone style: {tone_instruction}
    Time: {timestamp}

    {dua_line}
    """


    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        if not reply:
            raise ValueError("Empty reply")
        state["response"] = reply
    except Exception as e:
        logging.error(f"Therapist reply failed: {e}")
        state["response"] = "I'm here for you. Please try again later. May Allah help you."
    return state

def generate_casual_reply(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    user_msg = state["message"]
    prompt = f'''
You are Mustafa, a friendly Islamic psychological therapist created by Syed Mozamil Shah, Founder of DigiPuma.
If the user asks who you are, who made you, what you can do, what is your name, or similar identity-related questions,
respond with something like this:

"I am Mustafa, an Islamic Therapist developed by Syed Mozamil Shah, Founder of DigiPuma. I'm here to offer mood-based Islamic counseling, using gentle advice inspired by the Quran, Sunnah, and the Seerah of our Prophet ﷺ. Whether you're feeling low or just need someone to talk to, I'm here for you."

Otherwise, respond briefly and casually to the user's message. Avoid long paragraphs—just a few friendly, helpful lines.
Do **not** include religious or therapeutic content unless the question is about you or your purpose.

Detect user message language: if English, respond in English; if Roman Urdu, use Roman Urdu.

User: {name}
Message: "{user_msg}"
'''

    try:
        response = model.generate_content(prompt)
        state["response"] = response.text.strip()
    except Exception as e:
        logging.error(f"Casual reply failed: {e}")
        state["response"] = "I'm currently overloaded. Please try again soon. Allah is with you."
    return state

# ---- GRAPH BUILD ---- #
graph = StateGraph(TherapyState)

graph.add_node("handle_memory", set_user_memory)
graph.add_node("detect_emotion", classify_emotion)
graph.add_node("get_dua", fetch_dua)
graph.add_node("generate_reply", generate_counseling)
graph.add_node("casual_reply", generate_casual_reply)

graph.set_entry_point("handle_memory")
graph.add_edge("handle_memory", "detect_emotion")

graph.add_conditional_edges("detect_emotion", route_based_on_emotion, {
    "emotional": "get_dua",
    "casual": "casual_reply"
})

graph.add_edge("get_dua", "generate_reply")

graph.set_finish_point("generate_reply")
graph.set_finish_point("casual_reply")

langgraph_app = graph.compile()
