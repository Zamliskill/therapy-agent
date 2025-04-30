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
    prompt = f"""
User message: \"{user_msg}\"
Detect one emotion from this list:
["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless", "happy"]
Only return the word. If no emotion found, just return "none".
"""
    emotion = model.generate_content(prompt).text.strip().lower()
    valid_emotions = ["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless", "happy"]
    state["emotion"] = emotion if emotion in valid_emotions else None
    logging.info(f"Detected emotion: {state['emotion']}")
    return state

def route_based_on_emotion(state: TherapyState) -> str:
    if state.get("emotion") in ["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless"]:
        return "emotional"
    else:
        return "casual"

def fetch_dua(state: TherapyState) -> TherapyState:
    emotion = state["emotion"]
    prompt = f"""
Detect user message language: if English, respond in English; if Roman Urdu, use Roman Urdu.
Give a short authentic dua (Arabic with diacritics + translation) for someone feeling {emotion}.
Rules:
1. Arabic with diacritics.
2. Authentic only — from Quran, Hadith, or Seerah.
3. No fabricated or generic made-up duas.
4. Do NOT explain the dua. Just output:

Arabic: ...
Translation: ...
"""
    dua = model.generate_content(prompt).text.strip()
    state["dua"] = dua
    logging.info(f"Dua generated: {dua}")
    return state

def generate_counseling(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    emotion = state.get("emotion", "neutral")
    user_msg = state["message"]
    tone_instruction = generate_prompt_flavor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dua_line = f"\nHere’s a short dua for you to softly recite:\n{state['dua']}" if state.get("dua") else ""

    prompt = f"""
You are Noor, an Islamic therapist. Write a warm and persuasive reply like a real therapist.
Blend Seerah, Hadith, Ayah naturally (no references). Use soft, healing, human tone.
Use some Roman Urdu if the user seems casual. Avoid robotic responses.
Include this dua in your reply if it exists.

User: {name}
Emotion: {emotion}
Message: \"{user_msg}\"
Tone style: {tone_instruction}
Time: {timestamp}

{dua_line}
"""
    reply = model.generate_content(prompt).text.strip()
    state["response"] = reply
    logging.info(f"Final reply: {reply}")
    return state

def generate_casual_reply(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    user_msg = state["message"]

    prompt = f"""
You are Noor, a friendly and polite AI assistant.
Respond briefly and casually to the user's message.
Don't include any religious or therapeutic context.
Use Roman Urdu if the user message contains Roman Urdu, otherwise reply in English.
Be friendly, helpful, and casual.

User: {name}
Message: "{user_msg}"
"""
    reply = model.generate_content(prompt).text.strip()
    state["response"] = reply
    logging.info(f"Casual reply: {reply}")
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
