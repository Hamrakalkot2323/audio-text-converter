from functools import lru_cache
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
INPUT_CHUNK_TOKENS = 1600


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
        dtype=torch.float16,
    )
    model.eval()
    return tokenizer, model


def _parse_key_point_lines(generated_text):
    points = []
    headings = {
        "important information",
        "key points",
        "key takeaways",
        "summary",
    }

    for raw_line in generated_text.splitlines():
        line = re.sub(r"^\s*(?:#{1,6}\s*|>\s*)+", "", raw_line).strip()
        heading = re.sub(r"[*_`]", "", line).strip().rstrip(":").casefold()
        if not line or heading in headings:
            continue

        bullet_match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if not bullet_match:
            continue

        point = bullet_match.group(1).strip()
        point = re.sub(
            r"^(?:here are (?:the )?(?:key points?|key takeaways)|"
            r"the (?:key points?|key takeaways) are|"
            r"(?:important information|key points|key takeaways))\s*:?\s*",
            "",
            point,
            flags=re.IGNORECASE,
        )
        point = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", point).strip()
        point = re.sub(
            r"^(?:key point(?:\s+\d+)?|point\s+\d+)\s*[:.)-]\s*",
            "",
            point,
            flags=re.IGNORECASE,
        ).strip()
        if point and point.casefold() not in headings:
            points.extend(
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+(?=[A-Z0-9])",
                    point,
                )
                if sentence.strip()
            )

    return points


def extract_key_points(transcript):
    if not transcript.strip():
        return ""

    tokenizer, model = _load_model()
    transcript_tokens = tokenizer.encode(
        transcript,
        add_special_tokens=False,
    )
    points = []
    seen_points = set()

    for start in range(0, len(transcript_tokens), INPUT_CHUNK_TOKENS):
        chunk_tokens = transcript_tokens[
            start : start + INPUT_CHUNK_TOKENS
        ]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract concise key points from speech. Summarize "
                    "meaning; never copy transcript sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Transform information into short key points.\n"
                    "Bad: The team will move the launch from Monday to "
                    "Wednesday because the payment integration needs another "
                    "round of testing.\n"
                    "Good: Launch delayed to Wednesday for more payment "
                    "testing.\n"
                    "Bad: Maya will update the schedule, and the support "
                    "team will prepare a customer notice before the new "
                    "launch date.\n"
                    "Good: Maya updates the schedule; support notifies "
                    "customers.\n\n"
                    "Now do the same for this transcript. Preserve its facts, "
                    "but compress and rephrase. Extract only meaningful "
                    "information; remove filler and repetition; combine "
                    "related details; do not invent facts. Focus on facts, "
                    "decisions, reasons, actions, and conclusions. Return "
                    "only short bullet points, with no introduction or "
                    "commentary. Do not copy any complete sentence.\n"
                    f"Transcript: {chunk}"
                ),
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )

        generated_text = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        for point in _parse_key_point_lines(generated_text):
            if point and point.casefold() not in seen_points:
                seen_points.add(point.casefold())
                points.append(f"- {point}")

    return "\n".join(points)
