from enum import Enum


class TeamMode(str, Enum):
    """Competition team role for a simulation run."""

    RESCUE = "rescue"
    PLACE = "place"

    @classmethod
    def from_string(cls, value: str) -> "TeamMode":
        normalized = (value or cls.RESCUE.value).strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(f"Unknown team_mode: {value!r}. Use 'rescue' or 'place'.")
