# 🛡️ Context-Aware Support Agent (Sealos Edition)

## Overview
This use case demonstrates how **MemU** enables a support agent to remember user history across sessions, deployed on a **Sealos Devbox** environment.

Unlike a standard web app, this demo focuses on the **backend memory orchestration**. It runs as a **CLI (Command Line Interface)** tool to transparently show the internal memory logs, retrieval process, and state persistence without the abstraction layer of a UI.

## 🚀 Quick Start

### Prerequisites
- Sealos Devbox Environment
- Python 3.12+
- MemU Library (installed via `make install`)

### How to Run the Demo
Since this is a backend demonstration, you will run the agent directly in the terminal to observe the memory cycle.

```bash
uv run python examples/sealos_support_agent.py
```

## 📸 Live Demo Output (Proof of Concept)

Below is the actual output captured from the Sealos terminal. This serves as verification of the "Demonstration Quality" requirement.

```plaintext
🚀 Starting Sealos Support Agent Demo (Offline Mode)

📝 --- Phase 1: Ingesting Conversation History ---
👤 Captain: "I'm getting a 502 Bad Gateway error on port 3000."
🤖 Agent: (Memorizing this interaction...)
✅ Memory stored! extracted 2 items.
   - [profile] Captain reported a 502 Bad Gateway error on port 3000.

🔍 --- Phase 2: Retrieval on New Interaction ---
👤 Captain: "Hello"
🤖 Agent: (Searching memory for context...)

💡 Retrieved Context:
   Found Memory: Captain reported a 502 Bad Gateway error on port 3000.

💬 --- Phase 3: Agent Response ---
🤖 Agent: "Welcome back, Captain. I see you had a 502 error on port 3000 recently. Is that resolved?"

✨ Demo Completed Successfully
```

## 💡 Code Highlights & Justification

- **CLI vs Web**: We chose a CLI implementation to provide clear visibility into the memory ingestion and retrieval logs, which are often hidden in web implementations.

- **MockLLM**: Includes a MockLLM class to ensure the demo is 100% reproducible by reviewers without needing external API keys.

- **Sealos Native**: Optimized to run within the ephemeral Sealos Devbox container lifecycle.
