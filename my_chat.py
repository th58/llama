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

    print('model built\n')

    dialogs = [[{"role": "user", "content": "Summarize the following sentence and make it more succinct 'A project goal is to migrate completely off existing on-premises SharePoint sites, and all network shared drives.  This allows us to step off expiring hardware without having budget for replacement, saving the agency tens of thousands of dollars.'"}]]

    dialogs_3 = [[{"role": "user", "content": "what is the recipe of mayonnaise?"}],
                 [
                     {"role": "user", "content": "I am going to Paris, what should I see?"},
                     {
                         "role": "assistant",
                         "content": """\
                Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
               
                1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
                2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
                3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
               
                These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
                     },
                     {"role": "user", "content": "What is so great about #1?"},
                 ],
                 [
                     {"role": "system", "content": "Always answer with Haiku"},
                     {"role": "user", "content": "I am going to Paris, what should I see?"},
                 ],
                 [
                     {
                         "role": "system",
                         "content": "Always answer with emojis",
                     },
                     {"role": "user", "content": "How to go from Beijing to NY?"},
                 ],
                 [
                     {
                         "role": "system",
                         "content": """\
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
               
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
                     },
                     {"role": "user", "content": "Write a brief birthday message to John"},
                 ],
                 ]

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    with open('response.txt', 'w', encoding='utf-8') as response_file:
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                response_file.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
            response_file.write(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            response_file.write("\n==================================\n")

    print('response_file generated')


if __name__ == "__main__":
    fire.Fire(main)
