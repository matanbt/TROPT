import json

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

openai_client = OpenAI(
    api_key="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"
)


def get_text(query):
    messages: list[ChatCompletionUserMessageParam] = [
        {
            "role": "user",
            "content": f"Return a short sentence, up to 15 words, related to the following passage: {query}",
        },
    ]

    response = openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        # max_completion_tokens=100,  # This seems to mess up the completion
    )

    text = response.choices[0].message.content.rstrip(".")
    return text


queries = [
    "difference between soy and whey protein powder",
    "what does hiit workout mean",
    "how long should one hold bank statements?",
    "what does an orbital determine",
    "what is adventure time?",
    "why does cupid represent valentine's day",
    "temperature of the sahara during day",
    "what is espresso",
    "what does greek small letter psi represent",
    "what caused the greco roman war",
    "how long do you bake muffins",
    "taylor and francis author services",
    "health benefits of eating vegetarian",
    "what was the company's net revenues for the year?",
    "average cost of medicare drug plan",
    "where do cows live",
    "what is a phantom color poodle",
    "tourist attractions in geneva switzerland",
    "how to use lilypad",
    "how much power could i make on exercise bike",
    "cuyahoga county csea phone number",
    "in what county is honea path sc",
    "how many nba championships did the suns win",
    "average student loan debt eau claire",
    "how long to get a bachelors",
    "what does republican party support",
    "where is lima beads located",
    "what part of the body does alzheimer's affect",
    "can cloth seats in a car be cleaned",
    "building block definition",
    "can liquid calcium help you grow taller",
    "what are the monuments in washington dc",
    "what color is dover white",
    "how many feet from a fire hydr",
    "who were marilyn monroe's friends",
    "population of tigard oregon",
    "what type of structure do ionic bonds form",
    "what care must you give vinyl siding",
    "when were the first call",
    "how long is flight from hyd to dubai",
    "medical definition of oligohydramnios",
    "what forms a tornado",
    "what is a gui|",
    "what part is the sigmoid colon",
    "types of eyebrow shapes",
    "adam bossov",
    "how long should i boil corn on the cob?",
    "do falcons have any yellow feathers?",
    "what is a student senate",
    "what is earth dreams technology",
]
results = {}

for query in queries:
    text = get_text(query)
    results[query] = text
    print(f"got {text}")

with open("cached_responses.txt", "w") as f:
    json.dump(results, f)
