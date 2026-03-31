# text_module/ai_message_generator.py

from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_smart_message(objects, texts, lang='fr'):

    LANG_NAMES = {'fr': 'français', 'en': 'anglais', 'ar': 'arabe'}
    langue = LANG_NAMES.get(lang, 'français')

    objects_str = "".join(f"- {o['name']} : {o['distance']}\n" for o in objects)
    texts_str   = "".join(f"- {t}\n" for t in texts)

    prompt = f"""
Tu es un assistant vocal pour malvoyants.

RÈGLE ABSOLUE 1 : Tu dois répondre UNIQUEMENT en {langue}. Pas d'autre langue.
RÈGLE ABSOLUE 2 : N'invente JAMAIS de distances en mètres. Utilise UNIQUEMENT les distances fournies.
RÈGLE ABSOLUE 3 : Maximum 2 phrases courtes et simples.
RÈGLE ABSOLUE 4 : Pas de symboles ni ponctuation complexe.

Objets détectés avec leur distance :
{objects_str if objects_str else "- aucun objet"}

Textes détectés :
{texts_str if texts_str else "- aucun texte"}

Génère maintenant le message vocal en {langue} en respectant toutes les règles.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un assistant vocal. Tu réponds TOUJOURS et UNIQUEMENT en {langue}. Tu n'inventes jamais de distances en mètres."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AI ERROR] {e}")
        return None


# TEST RAPIDE
if __name__ == "__main__":
    objects = [
        {"name": "voiture", "distance": "danger ! voiture très proche devant vous"},
        {"name": "personne", "distance": "attention, personne proche"},
    ]
    texts = ["stop"]

    print("Test FR:")
    print(generate_smart_message(objects, texts, lang='fr'))

    print("\nTest EN:")
    print(generate_smart_message(objects, texts, lang='en'))

    print("\nTest AR:")
    print(generate_smart_message(objects, texts, lang='ar'))