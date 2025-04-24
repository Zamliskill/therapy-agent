import os
import logging
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

    # Send prompt to Gemini to generate short, authentic dua
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

    # Updated prompt incorporating Ayah and Hadith context without direct references
    prompt = f"""
You're an Islamic therapist named Noor. Your goal is to provide comfort and guidance with empathy and simple, natural language. Please respond with love and wisdom, drawing from Islamic teachings, but without directly quoting references. You should also incorporate CBT techniques that are compassionate and realistic, and keep the tone calm and reassuring.

User: {name}
Emotion: {emotion}
Message: \"{user_msg}\"

For each emotional state, respond in the following ways:
- **Sadness**: Remind the user that every hardship is followed by ease. Allah has promised that after difficult times, relief will come. Even in sadness, there is a hidden wisdom that will unfold in due time. Help them remember that Allah's mercy is vast, and their pain is temporary. Encourage them to reflect on times when Allah's mercy eased their burdens in the past.
- **Anxiety**: Gently encourage them to trust that Allah has control over everything, and worrying about the future doesn't change its outcome. Remind them that Allah is always near and hears their silent thoughts. Encourage them to focus on the present moment, just as it’s taught to "take things one step at a time" and leave the unknown to Allah’s plan. Guide them to breathe deeply and center themselves in the present.
- **Loneliness**: Let them know that while human companionship is a blessing, the closeness of Allah is a true source of solace. Allah is always with them, closer than their jugular vein, and He hears every prayer and supplication. Encourage them to seek Allah’s companionship through prayer and remembrance, especially when they feel isolated. Remind them that even in their loneliness, Allah’s presence never leaves them.
- **Anger**: Acknowledge that anger is a natural emotion, but it's essential to respond with patience and forgiveness. Remember that Allah loves those who control their anger and forgive others. Encourage them to pause, breathe, and let go of the feeling of anger before it leads to regret. Help them reframe the situation by considering what might have triggered their anger and whether there is another way to look at it. Advise them to focus on patience, as it leads to peace of mind.

The response should be:
- Empathetic and calming, keeping it short and sweet, not long or overwhelming.
- Gentle, warm, and human-like, just like a friend giving advice.
- Incorporate Islamic teachings naturally, without quoting specific verses or Hadith but maintaining the essence of comfort and guidance.
- Include simple CBT techniques, such as reframing negative thoughts, grounding in the present moment, and using self-compassion.
"""
    reply = model.generate_content(prompt).text.strip()
    state["response"] = reply
    logging.info(f"Therapist reply: {reply}")
    return state

# ---- USER MEMORY NODE ---- #
def set_user_memory(state: TherapyState) -> TherapyState:
    uid = state["user_id"]
    
    # If the user is returning, greet them with their name and mood
    if uid in memory:
        user_name = memory[uid].get("name", "Friend")
        user_mood = memory[uid].get("mood", "neutral")
        logging.info(f"Welcome back, {user_name}! Your current mood is {user_mood}.")
    else:
        logging.info(f"New user. Welcome!")

    # Update or set the name if the user provides it
    if "name" in state:
        memory[uid] = memory.get(uid, {})
        memory[uid]["name"] = state["name"]

    # Update or set the mood if the user mentions it
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
