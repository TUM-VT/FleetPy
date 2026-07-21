import json

from src.ml_gym.Actors import AbstractActor


class JSONWriter(AbstractActor):
    """Append every Hook observation to one JSON Lines file."""

    def __init__(self, output_f):
        self.output_f = output_f

    def compute_action(self, observation, process_id):
        with open(self.output_f, "a", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(
                    observation,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        return None
