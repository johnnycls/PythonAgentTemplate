from agent_template.cli_example.master.prompts import EXIT_COMMANDS
from agent_template.cli_example.master.agent import MasterAgent


def main():
    agent = MasterAgent()

    while True:
        user_input = input("You: ")
        if user_input.lower() in EXIT_COMMANDS:
            break
        response = agent.run([{"type": "text", "text": user_input}])
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
