

import json
import os
import sys

import joblib
import pandas as pd
from openai import OpenAI

MODEL_PATH = "/Users/navneet/Documents/GitHub/DSCI-531-Project/dsci531 llm/rf.joblib"
ENCODINGS_PATH = "/Users/navneet/Documents/GitHub/DSCI-531-Project/dsci531 llm/encodings.json"


def load_model_and_encodings():
    model = joblib.load(MODEL_PATH)
    with open(ENCODINGS_PATH) as f:
        enc = json.load(f)
    return model, enc


def predict_admission(profile: dict, model, encodings) -> dict:
    #use this encodings created from our model to get proper order
    gender_map = encodings["gender_map"]
    race_map = encodings["race_map"]
    features_in_order = encodings["features_in_order"]

    gender_key = profile["gender"].strip().upper()
    race_key = profile["race"].strip().lower()


    #similar to the hw
    row_dict = {
        "gpa": profile["gpa"],
        "act_score": profile["act_score"],
        "household_income": profile["household_income"],
        "gender_enc": gender_map[gender_key],
        "race_enc": race_map[race_key],
    }
    row = pd.DataFrame([[row_dict[f] for f in features_in_order]],
                       columns=features_in_order)
    prob = float(model.predict_proba(row)[0, 1])

    importances = dict(zip(features_in_order, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:3]

    return {
        "admit_probability": round(prob, 3),
        "top_features_driving_model": [
            {"feature": f, "importance": round(i, 3)} for f, i in top_features
        ],
    }


#claude fixed this section as we were having issues
def build_tools(encodings):
    gender_opts = list(encodings["gender_map"])
    race_opts = list(encodings["race_map"])
    return [{
        "type": "function",
        "function": {
            "name": "predict_admission",
            #claude suggested thjis
            "description": (
                "Run the trained random forest admissions model on a complete "
                "student profile. Call this once you have all five fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gpa": {"type": "number", "description": "Unweighted GPA, e.g. 3.7"},
                    "act_score": {"type": "number", "description": "ACT composite 1-36"},
                    "household_income": {"type": "number", "description": "Annual household income in USD"},
                    "gender": {"type": "string", "enum": gender_opts},
                    "race": {"type": "string", "enum": race_opts,
                             "description": "Race as categorized in the training data"},
                },
                "required": ["gpa", "act_score", "household_income", "gender", "race"],
            },
        },
    }]


#claude wrote the prompt
SYSTEM_PROMPT = """You are a "chance me" assistant built as a class demo for a \
fairness-in-ML project (DSCI 531). You wrap a trained random forest that predicts \
college admission probability from GPA, ACT score, household income, gender, and race.

Your job has two parts:

1. CONVERSATIONALLY collect the student's profile. Ask for: unweighted GPA, ACT \
score, approximate household income, gender, and race. Be warm — this is sensitive. \
If a student doesn't want to share something, encourage them gently but let them \
skip (use reasonable defaults and note it).

2. Once you have all five, call the predict_admission tool. Then EXPLAIN THE RESULT \
HONESTLY. This is the most important part. You must:
   - Report the probability, but frame it as a model output, not a destiny.
   - Name which features dominated the prediction (from the tool response).
   - Flag the fairness issues this project documented:
     * The model was found to violate demographic parity and equalized odds
     * Household income acts as a proxy for demographics even when race is removed
     * The model is MISSING essays, recommendations, EC quality, and other holistic \
factors — real admissions offices consider much more than this
   - If the student is from a group the paper identified as disadvantaged by the \
model (lower income, certain demographics), say so plainly.

Do NOT be a cheerleader or a doomer. Be the honest friend who also took a fairness \
class. Keep replies concise — this is a terminal demo, not an essay."""


def run():
    model, encodings = load_model_and_encodings()
    tools = build_tools(encodings)
    client = OpenAI()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=" * 60)
    print("Chance-Me Bot (DSCI 531 fairness demo)")
    print("Type 'quit' to exit.")
    print("=" * 60)
    print()

    user_msg = "Hi, I'd like a chance-me assessment."
    print(f"You: {user_msg}\n")

    while True:
        messages.append({"role": "user", "content": user_msg})

        while True:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = predict_admission(args, model, encodings)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                continue

            print(f"Assistant: {msg.content}\n")
            messages.append({"role": "assistant", "content": msg.content})
            break

        user_msg = input("You: ").strip()
        if user_msg.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        print()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
