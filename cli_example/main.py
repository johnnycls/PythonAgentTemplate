from threading import Event

from agent_template.cli_example.master.agent import MasterAgent

EXIT_COMMANDS = ("quit", "exit")


def main():
    agent = MasterAgent()
    abort = Event()

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in EXIT_COMMANDS:
                break
        except (KeyboardInterrupt, EOFError):
            print()
            continue
        abort.clear()
        
        try:
            for chunk in agent.run_stream([{"type": "text", "text": user_input}], abort=abort):
                if chunk.delta:
                    print(chunk.delta, end="", flush=True)
                elif chunk.tool_name:
                    print(f"\n[calling {chunk.tool_name}...]", end="", flush=True)
                elif chunk.tool_call:
                    print(f" -> {chunk.tool_call.name}({chunk.tool_call.arguments})")
        except KeyboardInterrupt:
            abort.set()
        abort.clear()


if __name__ == "__main__":
    main()
