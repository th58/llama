from typing import Optional
import fire
from llama import Llama


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    print('building model\n')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # This LLM is stateless, I need to keep a record of the previous conversations and continually add them back
    # into the dialog. However this can be limiting as we get restricted by the number of tokens we can parse.

    print('model built\n')
    # I need to put the loop to grab the prompt from the user
    user_prompt = input("What is your question? enter q to exit\n>   ")
    while user_prompt != 'q':
        # which puts it into the dialog
        dialogs = [[{"role": "user", "content": f"{user_prompt}"}]]
        # runs it through the chat_completion
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        # then will output to the console and prompt the user for more information.
        for result in results:
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            print("\n==================================\n")
        user_prompt = input("What is your question? enter q to exit\n>   ")


if __name__ == "__main__":
    fire.Fire(main)
