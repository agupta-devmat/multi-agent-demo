import logging
from strands import Agent
from strands.models import BedrockModel
from strands_tools.a2a_client import A2AClientToolProvider

from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables

# Grundlegendes Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Schritt 1: Definiere, wo der Supervisor zu finden ist ---
# Der Client muss nur die Adresse des Supervisors kennen.
SUPERVISOR_URL = "http://localhost:5000"


# --- Schritt 2: Den Tool Provider initialisieren ---
# Er wird sich mit dem Supervisor verbinden, seine Tools (call_search_agent, etc.)
# entdecken und daraus neue Tools für unseren Client-Agenten generieren.
logger.info(f"Versuche, Agenten unter {SUPERVISOR_URL} zu entdecken...")
try:
    provider = A2AClientToolProvider(known_agent_urls=[SUPERVISOR_URL])
except Exception as e:
    logger.error(f"Verbindung zum Supervisor Agent unter {SUPERVISOR_URL} fehlgeschlagen.")
    logger.error("Bitte stelle sicher, dass der supervisor_agent.py läuft.")
    logger.error(f"Fehlerdetails: {e}")
    exit()

# Überprüfen, ob Tools erfolgreich gefunden wurden
if not provider.tools:
    logger.error(f"Keine Tools vom Agenten unter {SUPERVISOR_URL} entdeckt.")
    logger.error("Stelle sicher, dass der Supervisor-Agent Tools definiert hat.")
    exit()

logger.info("Folgende Fähigkeiten vom Supervisor Agent erfolgreich entdeckt:")
for tool in provider.tools:
    # Der Name wird automatisch aus dem Tool-Namen des Supervisors generiert
    logger.info(f"- {tool.__name__}")


# --- Schritt 3: Einen lokalen Client-Agenten erstellen ---
# Dieser Agent läuft nur bei dir. Seine Aufgabe ist es, deine Anweisungen
# zu verstehen und die entdeckten Tools (also den Supervisor) zu nutzen.
client_bedrock_model = BedrockModel(
    model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0"
)

client_agent = Agent(
    system_prompt="""Du bist ein Kommandozeilen-Assistent. Deine Aufgabe ist es,
    die Anweisung des Benutzers zu analysieren und das am besten passende Werkzeug (Tool) auszuführen.
    Nutze die dir zur Verfügung stehenden Werkzeuge, um den Supervisor-Agenten zu finden und übertrage ihm die Aufgabe.""",
    tools=provider.tools,  # Hier werden die dynamisch erstellten Tools übergeben
    model=client_bedrock_model
)


# --- Schritt 4: Die interaktive Schleife ---
print("\n✅ Erfolgreich mit dem Supervisor verbunden.")
print("Du kannst jetzt komplexe, mehrstufige Aufgaben stellen.")
print("Beispiel: 'Search for the latest news on AI hardware and create a summary in a file named ai_news.md'")
print("Gib 'exit' ein, um das Programm zu beenden.")

while True:
    try:
        user_input = input("\n> ")
        if user_input.lower() == 'exit':
            break

        print("\n⏳ Denke nach und führe die Aufgabe aus...")
        
        # Der client_agent analysiert deine Eingabe und entscheidet,
        # welches entdeckte Tool er aufrufen muss.
        response = client_agent(user_input)

        print("\n--- Finale Antwort ---")
        print(response)
        print("----------------------")

    except KeyboardInterrupt:
        print("\nBeende Programm...")
        break
    except Exception as e:
        logger.error(f"Ein Fehler ist während der Konversation aufgetreten: {e}")
        break