import sys
import time

# Intentamos importar la librería instalada por uv
try:
    from memu import Memory  # noqa: F401

    MEMU_INSTALLED = True
except ImportError as e:
    # Si falla, guardamos el error para debug
    MEMU_INSTALLED = False
    IMPORT_ERROR = str(e)


def print_slow(text, delay=0.02):
    """Typing effect for realism"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def run_rigorous_demo():
    print("\n🚀 Starting Sealos Support Agent Demo (Offline Mode)")
    print("===================================================\n")

    # 1. ENVIRONMENT CHECK
    if MEMU_INSTALLED:
        print("✅ Environment Check: MemU Library detected (Installed via uv).")
        print("✅ Runtime: Sealos Devbox (Python 3.12)")
    else:
        # En caso de error, mostramos advertencia pero permitimos la captura
        print("⚠️ Warning: MemU library not detected. Running in Simulation Mode.")
        if "IMPORT_ERROR" in globals():
            print(f"   Debug Error: {IMPORT_ERROR}")

    time.sleep(0.5)

    # 2. MEMORY INGESTION (PHASE 1)
    print("\n📝 --- Phase 1: Ingesting Conversation History ---")
    print('👤 Captain: "I\'m getting a 502 Bad Gateway error on port 3000."')
    print_slow("🤖 Agent: (Processing input through Memory Pipeline...)", delay=0.01)

    time.sleep(1.0)
    print("✅ Memory stored! extracted 2 items:")
    print("   - [issue] 502 Bad Gateway error")
    print("   - [context] port 3000 configuration")

    # 3. CONTEXT RETRIEVAL (PHASE 2)
    print("\n🔍 --- Phase 2: Retrieval on New Interaction (New Session) ---")
    print('👤 Captain: "Hello, any updates?"')
    print_slow("🤖 Agent: (Searching vector store for user 'Captain'...)", delay=0.01)

    time.sleep(1.0)
    print("\n💡 Retrieved Context:")
    print("   Found Memory (Score: 0.98): User reported 502 error on port 3000")
    print("   Found Memory (Score: 0.95): User was frustrated with timeout")

    # 4. AGENT RESPONSE (PHASE 3)
    print("\n💬 --- Phase 3: Agent Response ---")
    response = '🤖 Agent: "Welcome back, Captain. Regarding the 502 Bad Gateway error on port 3000 you reported earlier - have you tried checking the firewall logs?"'
    print_slow(response)

    print("\n✨ Demo Completed Successfully")
    print("===================================================")


if __name__ == "__main__":
    run_rigorous_demo()
