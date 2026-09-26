"""AcademicDataPort: the isolation boundary between AI-QBE and whatever
academic database a given university already runs.

Per docs/PHASE0-DESIGN.md section 4.1 / master prompt section 5: AI-QBE
never assumes ownership of Program/Semester/Subject/Topic/SubTopic data.
Every place the rest of the codebase needs to know "what subjects/topics
exist" goes through this interface, never a direct query against a
university table. Swapping to a real institution's MySQL/MariaDB/Postgres
schema means writing one new adapter class here -- nothing in
backend/api, backend/generation, or backend/database/models.py changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectRef:
    external_id: str
    name: str
    program: str
    semester: str


@dataclass(frozen=True)
class TopicRef:
    external_id: str
    subject_external_id: str
    name: str
    subtopics: tuple[str, ...] = ()


class AcademicDataPort(abc.ABC):
    """Read-only interface onto the university's academic master data."""

    @abc.abstractmethod
    def list_subjects(self) -> list[SubjectRef]:
        ...

    @abc.abstractmethod
    def get_subject(self, external_subject_id: str) -> SubjectRef | None:
        ...

    @abc.abstractmethod
    def list_topics(self, external_subject_id: str) -> list[TopicRef]:
        ...

    @abc.abstractmethod
    def get_topic(self, external_subject_id: str, external_topic_id: str) -> TopicRef | None:
        ...


class InMemoryAcademicDataPort(AcademicDataPort):
    """Demo/dev/test adapter, seeded with the Physics-I example from the
    master prompt's business-objective section. NOT for production use --
    a real deployment implements AcademicDataPort against the university's
    actual SIS/academic database (see docs/PHASE0-DESIGN.md open question
    #1: confirm the target dialect, e.g. MySQL/MariaDB).
    """

    def __init__(self):
        self._subjects = {
            "phys-1": SubjectRef(
                external_id="phys-1", name="Physics-I", program="BS Engineering", semester="Semester 1"
            ),
        }
        self._topics = {
            "phys-1": [
                TopicRef("units-measurements", "phys-1", "Units and Measurements"),
                TopicRef("vectors", "phys-1", "Vectors", subtopics=("Dot Product", "Vector Addition")),
                TopicRef("motion", "phys-1", "Motion", subtopics=("Kinematics", "Free Fall")),
                TopicRef("newtons-laws", "phys-1", "Newton's Laws", subtopics=("Second Law", "Third Law")),
                TopicRef("work-energy", "phys-1", "Work and Energy"),
                TopicRef("circular-motion", "phys-1", "Circular Motion"),
                TopicRef("gravitation", "phys-1", "Gravitation"),
            ]
        }

    def list_subjects(self) -> list[SubjectRef]:
        return list(self._subjects.values())

    def get_subject(self, external_subject_id: str) -> SubjectRef | None:
        return self._subjects.get(external_subject_id)

    def list_topics(self, external_subject_id: str) -> list[TopicRef]:
        return list(self._topics.get(external_subject_id, []))

    def get_topic(self, external_subject_id: str, external_topic_id: str) -> TopicRef | None:
        for topic in self._topics.get(external_subject_id, []):
            if topic.external_id == external_topic_id:
                return topic
        return None
