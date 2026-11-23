from bisect import bisect_left, bisect_right
import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
from threading import Lock

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent
CORPUS_FILES = {
    "MultiTQ": ("MultiTQ/corpus.jsonl", "MultiTQ/e5_mean_Flat.index"),
    "CronQuestion": ("TimelineKGQA/CronQuestion/corpus.jsonl", "TimelineKGQA/CronQuestion/e5_mean_Flat.index"),
    "ICEWS": ("TimelineKGQA/ICEWS/corpus.jsonl", "TimelineKGQA/ICEWS/e5_mean_Flat.index"),
}

class UndefinedQuery(ValueError):
    pass


def period_bounds(value):
    if len(value) == 4:
        year = int(value)
        return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()
    if len(value) == 7:
        year, month = map(int, value.split("-"))
        last = calendar.monthrange(year, month)[1]
        return date(year, month, 1).isoformat(), date(year, month, last).isoformat()
    day = date.fromisoformat(value).isoformat()
    return day, day


@dataclass(frozen=True)
class Event:
    id: str
    contents: str
    subject: str
    predicate: str
    object: str
    date: str
    numeric_id: int


def load_events(corpus: Path) -> list[Event]:
    deduplicated: dict[tuple[str, str, str, str], Event] = {}
    with corpus.open() as source:
        for line in source:
            row = json.loads(line)
            event = Event(
                id=str(row["id"]),
                contents=row["contents"],
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                date=row["date"],
                numeric_id=int(row["id"]),
            )
            key = (event.subject, event.predicate, event.date, event.object)
            previous = deduplicated.get(key)
            if previous is None or event.numeric_id < previous.numeric_id:
                deduplicated[key] = event
    return sorted(
        deduplicated.values(),
        key=lambda event: (
            event.subject,
            event.predicate,
            event.date,
            event.numeric_id,
        ),
    )


class TemporalGraph:
    def __init__(self, events: list[Event]):
        self.events = events
        self.by_sp: dict[tuple[str, str], list[Event]] = defaultdict(list)
        self.by_po: dict[tuple[str, str], list[Event]] = defaultdict(list)
        for event in events:
            self.by_sp[event.subject, event.predicate].append(event)
            self.by_po[event.predicate, event.object].append(event)
        for value in {event.date for event in events}:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError(f"noncanonical corpus date: {value}")

class IntervalGraph:
    def __init__(self, corpus, path):
        self.corpus = corpus
        self.unit = "year" if corpus == "CronQuestion" else "day"
        self.path = Path(path)
        self.rows = {}
        self.intervals = {}
        self.by_sp = defaultdict(list)
        self.by_po = defaultdict(list)
        with self.path.open() as source:
            for line in source:
                row = json.loads(line)
                if row["start_time"] == row["end_time"]:
                    row = {**row, "contents": (f"{row['subject']} {row['predicate'].replace('_', ' ')} "
                                                f"{row['object']} on {row['start_time']}")}
                id_ = str(row["id"])
                if id_ in self.rows:
                    raise ValueError(f"duplicate occurrence ID: {id_}")
                self.rows[id_] = row
                self.intervals[id_] = (self.parse(row["start_time"]), self.parse(row["end_time"]))
                self.by_sp[row["subject"], row["predicate"]].append(id_)
                self.by_po[row["predicate"], row["object"]].append(id_)

    def parse(self, text):
        if text == "beginning of time":
            return -math.inf
        if text == "end of time":
            return math.inf
        # CronQuestion includes year zero.
        year, month, day = map(int, text.rsplit("-", 2))
        if self.unit == "year":
            return year
        return date(year, month, day).toordinal()

class GraphCorpus:
    def __init__(self, corpus, data_root=DEFAULT_DATA_ROOT):
        self.corpus, self.path = corpus, Path(data_root) / CORPUS_FILES[corpus][0]
        self.unit = "year" if corpus == "CronQuestion" else "day"
        if corpus == "MultiTQ":
            self.graph = TemporalGraph(load_events(self.path))
            self.rows = {e.id: {k: getattr(e, k) for k in
                         ("id", "contents", "subject", "predicate", "object", "date")}
                         for e in self.graph.events}
            self.intervals = {i: (date.fromisoformat(r["date"]).toordinal(),)*2 for i, r in self.rows.items()}
        else:
            self.graph = IntervalGraph(corpus, self.path)
            self.rows, self.intervals = self.graph.rows, self.graph.intervals

    def related_facts(self, keep, entity, predicate):
        matches = (self.graph.by_sp.get((entity, predicate), []) if keep == "subject"
                   else self.graph.by_po.get((predicate, entity), []))
        return [event.id for event in matches] if self.corpus == "MultiTQ" else list(matches)

    def period(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise UndefinedQuery("a_window_requires_two_endpoints")
            a, b = self.period(value[0])[0], self.period(value[1])[1]
        elif str(value).lower() in ("beginning of time", "end of time"):
            a = b = -math.inf if str(value).lower() == "beginning of time" else math.inf
        elif self.unit == "year":
            text = str(value)
            a = b = int(text.rsplit("-", 2)[0]) if len(text.rsplit("-", 2)) == 3 else int(text)
        else:
            lower, upper = period_bounds(str(value))
            a, b = date.fromisoformat(lower).toordinal(), date.fromisoformat(upper).toordinal()
        if a > b:
            raise UndefinedQuery("reversed_window")
        return a, b

    def format(self, value):
        if value in (-math.inf, math.inf):
            return "beginning of time" if value < 0 else "end of time"
        return str(int(value)) if self.unit == "year" else date.fromordinal(int(value)).isoformat()


@dataclass(frozen=True)
class Window:
    map: tuple[str, str, str]
    at: int
    period: tuple[int, int]
    ids: tuple[str, ...]
    before_hidden: int
    after_hidden: int
    total: int
    span: tuple[int, int]


class Timeline:
    def __init__(self, graph, window_size=15):
        self.graph = graph
        self.window_size = window_size
        self.maps = {}
        self._map_lock = Lock()
        self.aliases = {}
        if graph.corpus == "MultiTQ":
            by_fact = {tuple(r[k] for k in ("subject", "predicate", "object", "date")): i
                       for i, r in graph.rows.items()}
            with graph.path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    kept = by_fact[tuple(row[k] for k in ("subject", "predicate", "object", "date"))]
                    if str(row["id"]) != kept:
                        self.aliases[str(row["id"])] = kept

    def canonical(self, fact):
        if not isinstance(fact, str):
            raise ValueError("fact must be a string ID")
        fact = fact.removeprefix("F")
        fact = self.aliases.get(fact, fact)
        if fact not in self.graph.rows:
            raise ValueError(f"Unknown fact: {fact}")
        return fact

    def fact_line(self, fact):
        row = self.graph.rows[fact]
        if self.graph.corpus == "MultiTQ":
            when = row["date"]
        else:
            a, b = self.graph.intervals[fact]
            when = f"{self.graph.format(a)} to {self.graph.format(b)}"
        return f"F{fact}: {row['subject']} | {row['predicate']} | {row['object']} | {when}"

    def timeline(self, fact, keep, at, seen):
        fact = self.canonical(fact)
        if fact not in seen:
            raise ValueError("timeline fact must have been observed")
        if keep not in ("subject", "object"):
            raise ValueError("keep must be subject or object")
        row = self.graph.rows[fact]
        map_ = (row[keep], row["predicate"], keep)
        with self._map_lock:
            if map_ not in self.maps:
                ids = self.graph.related_facts(keep, row[keep], row["predicate"])
                ids.sort(key=lambda i: (self.graph.intervals[i][0], int(i)))
                dates = [self.graph.intervals[i][0] for i in ids]
                span = (dates[0], max(self.graph.intervals[i][1] for i in ids))
                self.maps[map_] = (ids, dates, span)
            ids, dates, span = self.maps[map_]
        if at is None:
            period = self.graph.intervals[fact]
        elif at in ("start", "end"):
            value = dates[0] if at == "start" else dates[-1]
            period = (value, value)
        else:
            if not isinstance(at, str):
                raise ValueError("at must be a calendar string, start or end")
            period = self.graph.period(at)
        a, b = period
        if self.graph.corpus == "MultiTQ":
            left, right = bisect_left(dates, a), bisect_right(dates, b)
            spare = max(0, self.window_size - (right - left))
            lo = max(0, left - spare // 2)
            hi = min(len(ids), right + spare - (left - lo))
            lo = max(0, lo - max(0, self.window_size - (hi - lo)))
            lo = bisect_left(dates, dates[lo])
            hi = bisect_right(dates, dates[hi - 1])
        else:
            overlaps = [index for index, id_ in enumerate(ids)
                        if self.graph.intervals[id_][0] <= b and self.graph.intervals[id_][1] >= a]
            if overlaps:
                anchor = ids.index(fact)
                center = anchor if anchor in overlaps else overlaps[len(overlaps) // 2]
            else:
                center = min(bisect_left(dates, a), len(ids) - 1)
            lo = max(0, center - self.window_size // 2)
            hi = min(len(ids), lo + self.window_size)
            lo = max(0, hi - self.window_size)
        return Window(map_, a, period, tuple(ids[lo:hi]), lo, len(ids) - hi,
                      len(ids), span)

    def render(self, window):
        entity, relation, keep = window.map
        fmt = self.graph.format
        lines = []
        at = self.at_text(window)
        lines.append(f"Timeline: {keep}={entity} | relation={relation} | at={at}"
                     f" | total={window.total} | span={fmt(window.span[0])} to {fmt(window.span[1])}"
                     f" | hidden before={window.before_hidden}, after={window.after_hidden}")
        a, b = window.period
        for fact in window.ids:
            start, end = self.graph.intervals[fact]
            mark = "before" if end < a else "after" if start > b else "same"
            lines.append(f"[{mark}] " + self.fact_line(fact))
        return "\n".join(lines)

    def at_text(self, window):
        lower, upper = map(self.graph.format, window.period)
        if lower == upper:
            return lower
        if self.graph.corpus != "MultiTQ":
            return f"{lower} to {upper}"
        return lower[:7] if lower[:7] == upper[:7] else lower[:4]
