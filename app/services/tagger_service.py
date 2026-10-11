import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.sessions import Session


def generate_caption(
    session: Session, index: int, tagger: str, prompt: str | None = None
) -> str:
    from lib.captioning import generate_caption as gen_caption

    state_dict = {"dataset": session.dataset}
    caption = gen_caption(index, tagger, state_dict, prompt=prompt)
    return caption or ""
