import os
import logging
import random
from datetime import datetime
from typing import TypedDict, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph

# ---- LOGGING SETUP ---- #
logging.basicConfig(level=logging.INFO)

# ---- STATE TYPE ---- #
class TherapyState(TypedDict, total=False):
    user_id: str
    name: Optional[str]
    message: str
    emotion: Optional[str]
    dua: Optional[str]
    response: Optional[str]

# ---- ENV + MODEL CONFIG ---- #
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is missing in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# ---- MEMORY STATE ---- #
memory = {}

# ---- DYNAMIC FLAVOR GENERATOR ---- #
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

# ---- EMOTION DETECTION NODE ---- #
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

# ---- DUA FETCH NODE ---- #
def fetch_dua(state: TherapyState) -> TherapyState:
    emotion = state["emotion"]

    prompt = f"""
Provide a short and authentic Islamic dua with proper diacritics (Arabic + English translation) for someone feeling {emotion}.
Keep the dua brief and concise, ensuring the translation is clear and meaningful.
Here are examples of correct format:
- For sadness: اللهم إني أعوذ بك من الهم والحزن - O Allah, I seek refuge in You from worry and grief.
- For anxiety: حَسْبُنَا اللَّهُ وَنِعْمَ الْوَكِيلُ - Allah is Sufficient for us, and He is the Best Disposer of affairs.
Format:
Arabic: ...
Translation: ...
"""
    dua = model.generate_content(prompt).text.strip()
    state["dua"] = dua
    logging.info(f"Dua provided: {dua}")
    return state

# ---- COUNSELOR RESPONSE NODE ---- #
def generate_counseling(state: TherapyState) -> TherapyState:
    name = state.get("name", "Friend")
    emotion = state["emotion"]
    user_msg = state["message"]

    tone_instruction = generate_prompt_flavor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""
You're an Islamic therapist named Noor. Respond as a compassionate friend blending Islamic wisdom with CBT coping skills.
Avoid references. Use context from Ayah/Hadith naturally. Make it feel like a warm, real conversation.

User: {name}
Emotion: {emotion}
Message: \"{user_msg}\"
Tone: {tone_instruction}
Time: {timestamp}

Guide for emotional response:
- Sad: Remind them that after hardship comes ease. Comfort them like light returns after darkness.
- Anxious: Encourage them to trust Allah’s plan. Help them ground in present.
- Lonely: Remind them Allah is closer than they think. They're never truly alone.
- Angry: Encourage calm, reframing, forgiveness. Point to the power in restraint.
- Others: Respond softly with emotional safety and spiritual insight.

Keep it short, empathetic, and human.
"""
    reply = model.generate_content(prompt).text.strip()
    state["response"] = reply
    logging.info(f"Therapist reply: {reply}")
    return state

# ---- USER MEMORY NODE ---- #
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

# ---- LANGGRAPH BUILD ---- #
graph = StateGraph(TherapyState)

graph.add_node("handle_memory", set_user_memory)
graph.add_node("detect_emotion", classify_emotion)
graph.add_node("get_dua", fetch_dua)
graph.add_node("generate_reply", generate_counseling)

graph.set_entry_point("handle_memory")
graph.add_edge("handle_memory", "detect_emotion")
graph.add_edge("detect_emotion", "get_dua")
graph.add_edge("get_dua", "generate_reply")
graph.set_finish_point("generate_reply")

langgraph_app = graph.compile()
