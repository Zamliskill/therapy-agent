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


# --- FALLBACK DUAS --- #
FALLBACK_DUAS = [
    {
        "arabic": "رَبِّ إِنِّي لِمَا أَنْزَلْتَ إِلَيَّ مِنْ خَيْرٍ فَقِيرٌ",
        "translation": "My Lord, indeed I am in need of whatever good You would send down to me."
    },
    {
        "arabic": "اللّهُمَّ إِنِّي أَسْأَلُكَ رِضَاكَ وَالجَنَّةَ وَأَعُوذُ بِكَ مِنْ سَخَطِكَ وَالنَّارِ",
        "translation": "O Allah, I ask You for Your pleasure and Paradise, and I seek refuge in You from Your anger and the Fire."
    },
    {
        "arabic": "اللّهُمَّ لا تَجْعَلْ مُصِيبَتَنَا فِي دِينِنَا",
        "translation": "O Allah, do not make our affliction in our religion."
    },
    {
        "arabic": "اللّهُمَّ ثَبِّتْ قَلْبِي عَلَى دِينِكَ",
        "translation": "O Allah, make my heart steadfast upon Your religion."
    },
    {
        "arabic": "رَبَّنَا آتِنَا فِي الدُّنْيَا حَسَنَةً وَفِي الآخِرَةِ حَسَنَةً وَقِنَا عَذَابَ النَّارِ",
        "translation": "Our Lord, give us in this world [that which is] good and in the Hereafter [that which is] good, and protect us from the punishment of the Fire."
    }
]

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


# --- DUA DATASET --- #
DUA_DATASET = {
    "sad": [
        {
            "arabic": "اللّهُمَّ إِنِّي أَعُوذُ بِكَ مِنَ الهَمِّ وَالحَزَنِ",
            "translation": "O Allah, I seek refuge in You from worry and grief."
        },
        {
            "arabic": "اللّهُمَّ اجْبُرْ كَسْرِي وَارْزُقْنِي الرِّضَا",
            "translation": "O Allah, mend my brokenness and grant me contentment."
        },
        {
            "arabic": "اللّهُمَّ املأ قلبي سرورًا وأملاً بك",
            "translation": "O Allah, fill my heart with joy and hope in You."
        },
        {
            "arabic": "اللّهُمَّ إِنِّي أَسْأَلُكَ نَفْسًا مُطْمَئِنَّةً",
            "translation": "O Allah, I ask You for a soul that is content."
        },
        {
            "arabic": "اللّهُمَّ اجعلني ممن تبشرهم الملائكة: ألا تخافوا ولا تحزنوا",
            "translation": "O Allah, make me among those whom the angels give glad tidings: 'Do not fear and do not grieve.'"
        }
    ],
    "anxious": [
        {
            "arabic": "اللّهُمَّ لاَ سَهْلَ إِلاَّ مَا جَعَلْتَهُ سَهْلاً",
            "translation": "O Allah, there is no ease except what You make easy."
        },
        {
            "arabic": "اللّهُمَّ اكفني همي وأزل عني كربي",
            "translation": "O Allah, relieve me of my worry and remove my distress."
        },
        {
            "arabic": "اللّهُمَّ طمئن قلبي بذكرك",
            "translation": "O Allah, reassure my heart with Your remembrance."
        },
        {
            "arabic": "اللّهُمَّ إني أعوذ بك من الهم والحزن",
            "translation": "O Allah, I seek refuge in You from anxiety and sorrow."
        },
        {
            "arabic": "اللّهُمَّ اشرح لي صدري ويسر لي أمري",
            "translation": "O Allah, expand for me my chest and ease for me my task."
        }
    ],
    "hopeless": [
        {
            "arabic": "اللّهُمَّ ارزقني حسن الظن بك",
            "translation": "O Allah, grant me good thoughts about You."
        },
        {
            "arabic": "اللّهُمَّ اجعلني من المتوكلين عليك",
            "translation": "O Allah, make me among those who rely upon You."
        },
        {
            "arabic": "اللّهُمَّ لا تحرمني خير ما عندك بسوء ما عندي",
            "translation": "O Allah, do not deprive me of the best of what You have because of the worst of what I have."
        },
        {
            "arabic": "رَبِّ لَا تَذَرْنِي فَرْدًا وَأَنتَ خَيْرُ الْوَارِثِينَ",
            "translation": "My Lord, do not leave me alone, and You are the best of inheritors."
        },
        {
            "arabic": "اللّهُمَّ اجعل آخر كلامي شهادة أن لا إله إلا الله",
            "translation": "O Allah, make the last words I speak: There is no god but Allah."
        }
    ],
    "guilty": [
        {
            "arabic": "اللّهُمَّ إِنِّي ظَلَمْتُ نَفْسِي ظُلْمًا كَثِيرًا فَاغْفِرْ لِي",
            "translation": "O Allah, I have greatly wronged myself, so forgive me."
        },
        {
            "arabic": "رَبِّ اغْفِرْ وَارْحَمْ وَأَنتَ خَيْرُ الرَّاحِمِينَ",
            "translation": "My Lord, forgive and have mercy, and You are the best of the merciful."
        },
        {
            "arabic": "اللّهُمَّ اجعلني من التوابين",
            "translation": "O Allah, make me among those who repent often."
        },
        {
            "arabic": "اللّهُمَّ طَهِّرْ قلبي من الذنوب والخطايا",
            "translation": "O Allah, purify my heart from sins and mistakes."
        },
        {
            "arabic": "رَبَّنَا اغْفِرْ لَنَا ذُنُوبَنَا وَكَفِّرْ عَنَّا سَيِّئَاتِنَا",
            "translation": "Our Lord, forgive us our sins and remove from us our misdeeds."
        }
    ],
    "lonely": [
        {
            "arabic": "اللّهُمَّ آنِسْ وَحْشَتِي",
            "translation": "O Allah, comfort my loneliness."
        },
        {
            "arabic": "اللّهُمَّ اكفني بحلالك عن حرامك وأغنني بفضلك عمن سواك",
            "translation": "O Allah, suffice me with Your lawful against Your unlawful, and enrich me by Your bounty over all besides You."
        },
        {
            "arabic": "اللّهُمَّ كُنْ مَعِي وَلاَ تَكُنْ عَلَيَّ",
            "translation": "O Allah, be with me and not against me."
        },
        {
            "arabic": "اللّهُمَّ إني أسألك أنس القلب بقربك",
            "translation": "O Allah, I ask You for the comfort of the heart through closeness to You."
        },
        {
            "arabic": "اللّهُمَّ املأ قلبي بنورك ورضاك",
            "translation": "O Allah, fill my heart with Your light and Your pleasure."
        }
    ]
}

# --- DUA FETCH NODE --- #
def fetch_dua(state: TherapyState) -> TherapyState:
    emotion = state.get("emotion")

    # 🛑 NEW ADDITION: If emotion is none/neutral/happy, no dua needed
    if emotion in ["none", "neutral"]:
        state["dua"] = None
        return state

    if emotion in DUA_DATASET:
        selected_dua = random.choice(DUA_DATASET[emotion])
        dua_text = f"Arabic: {selected_dua['arabic']}\nTranslation: {selected_dua['translation']}"
        state["dua"] = dua_text
        return state

    selected_dua = random.choice(FALLBACK_DUAS)
    dua_text = f"Arabic: {selected_dua['arabic']}\nTranslation: {selected_dua['translation']}"
    state["dua"] = dua_text
    return state




# --- COUNSELOR RESPONSE NODE --- #
def generate_counseling(state: TherapyState) -> TherapyState:
    include_dua = bool(state.get("dua"))

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
- - If the emotion is sad, anxious, lonely, guilty, hopeless, angry, tired, or empty: end with a heart-touching dua with proper arabic diacritics and translation.
- If the emotion is neutral, or positive, no need to include any dua. Just talk naturally.
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

graph.add_conditional_edges(
    "detect_emotion",
    lambda state: "get_dua" if state.get("emotion") in ["sad", "angry", "anxious", "tired", "lonely", "guilty", "empty", "hopeless"] else "generate_reply"
)

graph.add_edge("get_dua", "generate_reply")
graph.set_finish_point("generate_reply")

langgraph_app = graph.compile()
