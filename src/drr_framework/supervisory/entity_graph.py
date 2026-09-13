"""Sourced legal-entity relationships. Overlapping perimeters are not summed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .common import day, instant, source_url, stable_id


@dataclass(frozen=True)
class PublicEntity:
    identifier: str
    name: str
    entity_type: str
    jurisdiction: str
    source: str
    available_as_of: str
    rssd: Optional[str] = None
    provenance: str = "public_documentation"

    def __post_init__(self):
        if self.provenance != "synthetic_test":
            source_url(self.source)
        if self.entity_type not in {
            "foreign_parent",
            "combined_us_operations",
            "ihc",
            "bank",
            "nonbank",
            "branch",
            "agency",
            "other",
        }:
            raise ValueError("Unknown entity type")
        instant(self.available_as_of)


@dataclass(frozen=True)
class EntityRelationship:
    parent: str
    child: str
    relationship: str
    effective_from: str
    available_as_of: str
    source: str
    effective_to: Optional[str] = None
    provenance: str = "public_documentation"

    def __post_init__(self):
        if self.parent == self.child:
            raise ValueError("Self relationships are invalid")
        if self.provenance != "synthetic_test":
            source_url(self.source)
        day(self.effective_from)
        instant(self.available_as_of)
        if self.effective_to and day(self.effective_to) < day(self.effective_from):
            raise ValueError("Relationship effective dates are reversed")

    @property
    def relationship_id(self):
        return stable_id(self)


class EntityGraph:
    def __init__(self, entities=(), relationships=()):
        self.entities = tuple(entities)
        self.relationships = tuple(relationships)
        identifiers = {e.identifier for e in entities}
        if len(identifiers) != len(self.entities):
            raise ValueError("Duplicate entity identifier")
        if any(r.parent not in identifiers or r.child not in identifiers for r in relationships):
            raise ValueError("Relationship references undocumented entity")

    def active(self, *, as_of, effective_on=None):
        effective_on = effective_on or as_of
        ids = {e.identifier for e in self.entities if instant(e.available_as_of) <= instant(as_of)}
        rows = tuple(
            r
            for r in self.relationships
            if r.parent in ids
            and r.child in ids
            and instant(r.available_as_of) <= instant(as_of)
            and day(r.effective_from) <= day(effective_on)
            and (r.effective_to is None or day(effective_on) <= day(r.effective_to))
        )
        import networkx as nx

        graph = nx.DiGraph((r.parent, r.child) for r in rows)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Conflicting/cyclic entity perimeter requires analyst review")
        return rows

    def _walk(self, identifier, *, as_of, effective_on=None, reverse=False):
        rows = self.active(as_of=as_of, effective_on=effective_on)
        found = set()
        pending = [identifier]
        while pending:
            current = pending.pop()
            for r in rows:
                parent, child = (r.child, r.parent) if reverse else (r.parent, r.child)
                if parent == current and child not in found:
                    found.add(child)
                    pending.append(child)
        return tuple(sorted(found))

    def ancestors(self, identifier, **kwargs):
        return self._walk(identifier, reverse=True, **kwargs)

    def descendants(self, identifier, **kwargs):
        return self._walk(identifier, **kwargs)

    def us_perimeter(self, identifier, **kwargs):
        ids = set(self.descendants(identifier, **kwargs)) | {identifier}
        return tuple(
            sorted(
                e.identifier
                for e in self.entities
                if e.identifier in ids
                and e.jurisdiction == "US"
                and instant(e.available_as_of) <= instant(kwargs["as_of"])
            )
        )

    def ihc_perimeter(self, identifier, **kwargs):
        descendants = set(self.descendants(identifier, **kwargs)) | {identifier}
        ihcs = [
            e.identifier
            for e in self.entities
            if e.identifier in descendants
            and e.entity_type == "ihc"
            and instant(e.available_as_of) <= instant(kwargs["as_of"])
        ]
        return tuple(sorted(set(ihcs).union(*(set(self.descendants(i, **kwargs)) for i in ihcs))))

    def branch_agency_perimeter(self, identifier, **kwargs):
        ids = set(self.us_perimeter(identifier, **kwargs))
        return tuple(
            sorted(
                e.identifier
                for e in self.entities
                if e.identifier in ids and e.entity_type in {"branch", "agency"}
            )
        )

    def overlays(self, identifier, values, *, as_of):
        ids = set(self.us_perimeter(identifier, as_of=as_of))
        return {key: value for key, value in values.items() if key in ids}
