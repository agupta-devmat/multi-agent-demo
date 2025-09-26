import logging
from strands import Agent
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer
from strands_tools.a2a_client import A2AClientToolProvider # Wichtigster Import!

logging.getLogger("strands").setLevel(logging.INFO)

from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables

# --- Schritt 1: Definiere, wo die Spezialisten-Agenten zu finden sind ---
SPECIALIST_AGENT_URLS = [
    "http://localhost:5001",  # Search Agent
    "http://localhost:5002",  # Markdown Agent
]


# --- Schritt 2: Den Tool Provider initialisieren, um die Spezialisten zu entdecken ---
print("Supervisor: Entdecke Spezialisten-Agenten...")
try:
    provider = A2AClientToolProvider(known_agent_urls=SPECIALIST_AGENT_URLS)
except Exception as e:
    print(f"Supervisor: Fehler beim Entdecken der Agenten: {e}")
    exit()

if not provider.tools:
    print("Supervisor: Keine Tools von Spezialisten-Agenten entdeckt. Stelle sicher, dass sie laufen.")
    exit()

print("Supervisor: Folgende Fähigkeiten von Spezialisten erfolgreich entdeckt:")
for tool in provider.tools:
    # Der Tool-Name wird automatisch generiert, z.B. 'search_agent_websearch'
    print(f"- {tool.__name__}")


# --- Schritt 3: Den Supervisor Agenten mit den dynamisch entdeckten Tools definieren ---

bedrock_model = BedrockModel(
    model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0"
)

# Der System-Prompt wird angepasst, um die neuen, dynamischen Tools zu nutzen
supervisor_agent = Agent(
    name="Supervisor Agent",
    description="Ein Supervisor-Agent, der andere spezialisierte Agenten orchestriert.",
    system_prompt="""Du bist ein intelligenter Supervisor-Agent.
    Deine Aufgabe ist es, die Anfragen des Benutzers zu analysieren und sie in logische Schritte zu zerlegen.
    Benutze deine Werkzeuge um den richten Agenten für die jeweilige Aufgabe zu finden.
    Wähle das passende Werkzeug des richtigen Spezialisten, um die Aufgabe zu erfüllen.
    Kombiniere die Werkzeuge, um mehrstufige Aufgaben zu lösen.""",
    
    
    # Die Tools des Supervisors sind jetzt die dynamisch erstellten Proxy-Tools
    tools=provider.tools,
    
    model=bedrock_model,
)


# --- Schritt 4: Den Supervisor auf seinem Port starten ---

print("Supervisor Agent wird gestartet...")
a2a_server = A2AServer(
    agent=supervisor_agent,
    http_url="http://localhost:5000"
)
a2a_server.serve(host="0.0.0.0", port=5000)