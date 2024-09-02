import time

from openai import OpenAI

client = OpenAI(
    # curl http://34.32.195.175:8000
    # base_url="http://localhost:8000/v1",
    base_url="http://34.34.94.211:8000/v1",
    api_key="token_abc123",
)


def calculate_tokens_per_second(num_tokens, elapsed_time):
    return num_tokens / elapsed_time if elapsed_time > 0 else 0


def stream_chat_completion(messages):
    start_time = time.time()
    full_response = ""

    # model="felarof01/test-llama3.1",
    stream = client.chat.completions.create(
        model=
        "felarof01/test-llama3.1-8b-instruct",  # "felafax/llama-3.1-8B-Instruct-JAX",
        messages=messages,
        stream=True,
    )

    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Estimate token count (this is a rough estimate, actual token count may vary)
    estimated_tokens = len(full_response.split())
    tokens_per_second = calculate_tokens_per_second(estimated_tokens,
                                                    elapsed_time)

    print(f"\n\nTokens/s: {tokens_per_second:.2f}")
    return full_response


def main():
    messages = []
    print("Welcome to the chat! Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        assistant_response = stream_chat_completion(messages)
        messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
