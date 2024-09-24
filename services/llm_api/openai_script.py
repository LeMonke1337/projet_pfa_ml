from openai import OpenAI

client = OpenAI()

client.api_key = "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"


async def is_gibberish(text):
    messages = [
        {
            "role": "system",
            "content": "Is the following text contains alot of gibberish words with no meaning?'Yes' or 'No'",
        },
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=messages, max_tokens=4, temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer == "yes"


async def is_enough(text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a career recommendation assistant. Only respond 'Yes' or 'No'. "
                "Use the following rules to evaluate the text: "
                "1. Experience must be relevant to a job or profession. "
                "2. Education must be more specific than 'bachelor' and ideally related to the target occupation. "
                "3. Skills must include professional or technical skills, not just generic skills like 'driving license' or 'basic computer skills'. "
                "Evaluate if the given information is enough for a valid occupation recommendation."
            ),
        },
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=messages, max_tokens=4, temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer == "yes"


async def get_explanation(text, result):
    messages = [
        {
            "role": "system",
            "content": f"with the following text : {text} , and result : {result} from prediction , give me an explanation of why and make it not too long",
        },
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=messages, max_tokens=255, temperature=0
    )

    answer = response.choices[0].message.content

    return answer


async def get_specifications(text, result):
    messages = [
        {
            "role": "system",
            "content": f"with the following text : {text} , and result : {result} from prediction , give me three specialities of said occupation , only names no explanation ",
        },
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=messages, max_tokens=40, temperature=0
    )

    answer = response.choices[0].message.content

    return answer
