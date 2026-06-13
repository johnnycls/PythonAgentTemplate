from agent_template.cli_example.master.prompts import EXIT_COMMANDS
from agent_template.cli_example.master.agent import MasterAgent


def main():
    agent = MasterAgent()

    while True:
        user_input = input("You: ")
        if user_input.lower() in EXIT_COMMANDS:
            break
        for chunk in agent.run_stream([{"type": "text", "text": user_input}]):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            elif chunk.tool_name:
                print(f"\n[{chunk.tool_name}...]")
        print()


if __name__ == "__main__":
    main()

