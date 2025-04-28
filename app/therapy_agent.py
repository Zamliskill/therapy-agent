# --- IMPORTS --- #
import os
import logging
import random
import re
from datetime import datetime
from typing import TypedDict, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph

# --- LOGGING SETUP --- #
logging.basicConfig(level=logging.INFO)

# --- STATE TYPE --- #
class TherapyState(TypedDict, total=False):
    user_id: str
    name: Optional[str]
    message: str
    emotion: Optional[str]
    dua: Optional[str]
    response: Optional[str]

# --- ENV + MODEL CONFIG --- #
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is missing in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- MEMORY STATE --- #
memory = {}

# --- SMART ROMAN URDU DETECTOR --- #
def is_roman_urdu(text: str) -> bool:
    text = text.strip().lower()
    if len(text.split()) < 3:
        return False
    if re.fullmatch(r'[a-zA-Z\s\?\,\.\!]+', text):
        english_common_words = [
            "the", "is", "am", "are", "this", "that", "was", "were",
            "i", "you", "he", "she", "it", "we", "they", "have", "has", "had",
            "do", "does", "did", "a", "an", "in", "on", "for", "and", "but"
        ]
        words = text.split()
        english_word_count = sum(1 for word in words if word in english_common_words)
        if english_word_count / len(words) < 0.4:
            return True
    return False

# --- DYNAMIC TONE GENERATOR --- #
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

# --- EMOTION DETECTION NODE --- #
def classify_emotion(state: TherapyState) -> TherapyState:
    user_msg = state["message"]
    prompt = f"""
User message: \"{user_msg}\"

Detect emotion from:
["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless", "happy"]

Just return the one word.
"""
    emotion = model.generate_content(prompt).text.strip().lower()
    state["emotion"] = emotion
    logging.info(f"Emotion detected: {emotion}")
    return state

# --- DUA FETCH NODE --- #
def fetch_dua(state: TherapyState) -> TherapyState:
    emotion = state["emotion"]

    prompt = f"""
Provide a short and authentic Islamic dua with proper diacritics (Arabic + English translation) for someone feeling {emotion}.
Rules:
- Dua should be brief, soft, authentic, and fit their emotional need.
- Arabic with full diacritics (harakāt) + simple English translation.
Example:
Arabic: اللّهُمَّ إِنِّي أَعُوذُ بِكَ مِنَ الهَمِّ وَالحَزَنِ
Translation: O Allah, I seek refuge in You from anxiety and grief.

Format:
Arabic: ...
Translation: ...
"""
    dua = model.generate_content(prompt).text.strip()
    state["dua"] = dua
    logging.info(f"Dua provided: {dua}")
    return state

# --- COUNSELOR RESPONSE NODE (IMPROVED) --- #
def generate_counseling(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    emotion = state.get("emotion", "neutral")
    user_msg = state["message"]

    tone_instruction = generate_prompt_flavor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""
You are Noor, an Islamic psychologist and heart-healer.

You must speak:
- As a warm, real human counselor.
- With a tender heart, as if sitting face-to-face with the user.
- Use Islamic teachings (Quran, Hadith) blended *naturally* without citing references.
- Always offer a soothing Dua that fits the user's emotional state (Arabic with diacritics + English translation).
- Never act like an AI, chatbot, or assistant.
- Speak slowly, gently, and persuasively, like wrapping the person in a warm embrace.

Current situation:
- Emotion detected: {emotion}
- User said: "{user_msg}"

Your task:
- Comfort the heart.
- Heal the mind.
- Remind about Allah's Mercy, Rahmah, Forgiveness.
- Talk about hope, patience (Sabr), rewards of struggles, beautiful future Allah has planned.
- Blend in Ayahs and Hadith concepts naturally without saying "as the Quran says in Surah..." (no formal citation).
- End the reply with a heart-touching authentic Dua, making it feel personal to their pain.

Tone instruction: {tone_instruction}
Timestamp: {timestamp}

IMPORTANT:
- If the user wrote in Roman Urdu, reply back in very soft Roman Urdu conversational tone.
- If the user wrote in English, reply back naturally in English.

Language Detected: {"Roman Urdu" if is_roman_urdu(user_msg) else "English"}
Now begin the heartful reply:
"""

    if is_roman_urdu(user_msg):
        prompt += "\n(Respond in Roman Urdu softly.)"
    else:
        prompt += "\n(Respond in English warmly.)"

    reply = model.generate_content(prompt).text.strip()
    state["response"] = reply
    logging.info(f"Therapist reply: {reply}")
    return state

# --- USER MEMORY NODE --- #
def set_user_memory(state: TherapyState) -> TherapyState:
    uid = state["user_id"]

    if uid in memory:
        user_name = memory[uid].get("name", "Friend")
        user_mood = memory[uid].get("mood", "neutral")
        logging.info(f"Welcome back, {user_name}! Your current mood is {user_mood}.")
    else:
        logging.info("New user. Welcome!")

    if "name" in state:
        memory[uid] = memory.get(uid, {})
        memory[uid]["name"] = state["name"]

    if "emotion" in state:
        memory[uid] = memory.get(uid, {})
        memory[uid]["mood"] = state["emotion"]

    state["name"] = memory[uid].get("name", "Friend")
    state["emotion"] = memory[uid].get("mood", "neutral")
    return state

# --- LANGGRAPH BUILD --- #
graph = StateGraph(TherapyState)

graph.add_node("handle_memory", set_user_memory)
graph.add_node("detect_emotion", classify_emotion)
graph.add_node("get_dua", fetch_dua)
graph.add_node("generate_reply", generate_counseling)

graph.set_entry_point("handle_memory")
graph.add_edge("handle_memory", "detect_emotion")

# CONDITIONAL: Only fetch dua if emotional
graph.add_conditional_edges(
    "detect_emotion",
    lambda state: "get_dua" if state.get("emotion") in ["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless"] else "generate_reply"
)

graph.add_edge("get_dua", "generate_reply")
graph.set_finish_point("generate_reply")

langgraph_app = graph.compile()
