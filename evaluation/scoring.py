"""Offline scoring for MultiTQ and TimelineKGQA predictions."""
import calendar
import math
import re
import unicodedata
from datetime import date, datetime, timedelta
from decimal import Decimal


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def endpoint(value, corpus):
    text = str(value).strip()
    if text.lower() in ("beginning of time", "end of time"):
        return -math.inf if text.lower() == "beginning of time" else math.inf
    if corpus == "CronQuestion":
        if not re.fullmatch(r"-?\d+", text):
            raise ValueError("expected a recorded year")
        return int(text)
    return date(*map(int, text.split("-"))).toordinal()


def answer_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected a nonempty name')
    text = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', text).split())


def time_text(value, level):
    text = str(value).strip().strip('\"\'').strip()
    if text.lower() in ('beginning of time', 'end of time'):
        return text.lower()
    if text.lower().startswith('0001-01-01t'):
        return 'beginning of time'
    if text.lower().startswith('9999-12-31t'):
        return 'end of time'
    if level == 'year' and re.fullmatch(r'-?\d+', text):
        return str(int(text))
    iso = re.split('[Tt]', text)[0].replace('/', '-').replace('–', '-').replace('−', '-')
    match = re.fullmatch(r'(-?\d+)(?:-(\d{1,2})(?:-(\d{1,2}))?)?', iso)
    if match:
        year, month, day = match.groups()
        y = int(year)
        if month is not None and not 1 <= int(month) <= 12:
            raise ValueError('invalid month')
        if day is not None:
            date(y, int(month), int(day))
        if level == 'year':
            return str(y)
        if level == 'month' and month is not None:
            return f'{y:04d}-{int(month):02d}'
        if level == 'day' and day is not None:
            return f'{y:04d}-{int(month):02d}-{int(day):02d}'
        raise ValueError('answer has less precision than requested')
    for pattern in ('%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y', '%B %Y', '%b %Y'):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if level == 'day' and '%d' not in pattern:
            raise ValueError('answer has less precision than requested')
        return time_text(parsed.strftime('%Y-%m-%d'), level)
    raise ValueError('unrecognized explicit time')


def interval_answer(value, corpus):
    empty_markers = ('no answer', 'no overlap', 'no common period', 'empty', '[]', 'null', 'none')
    marker = value
    while isinstance(marker, list) and len(marker) == 1:
        marker = marker[0]
    if isinstance(marker, str) and marker.strip().casefold() in empty_markers:
        return []
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        value = value[0]
    if value is None or value == []:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in empty_markers:
            return []
        text = text.strip('()[]')
        parts = re.split(r'\s+(?:to|and|[-–—])\s+|,\s*', text, flags=re.I)
        if len(parts) == 1:
            compact = re.fullmatch(r'(\d{4})\s*[-–—]\s*(\d{4})', text)
            parts = list(compact.groups()) if compact else parts
        value = parts
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (str, int)) for x in value):
        value = [value]
    if not isinstance(value, list):
        raise ValueError('expected interval endpoint pairs')
    level = 'year' if corpus == 'CronQuestion' else 'day'
    result = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('expected interval endpoint pairs')
        result.append([time_text(x, level) for x in pair])
    bounds = [(endpoint(a, corpus), endpoint(b, corpus)) for a, b in result]
    if any(a > b for a, b in bounds):
        raise ValueError("reversed predicted interval")
    if any(current[0] <= previous[1] + 1 for previous, current in zip(bounds, bounds[1:])):
        raise ValueError("intervals must already be ordered, disjoint and maximal; merging is not performed")
    return result


def duration_seconds(value):
    text = str(value).strip().strip('()').casefold()
    if text in ('forever', 'infinite', 'infinity', 'infinite duration', 'endless',
                'indefinite', 'indefinite duration'):
        return 'forever'
    span_text = re.sub(r'^from\s+', '', text)
    dates = re.split(r'\s+-\s+|\s+to\s+|,\s*', span_text)
    if len(dates) == 2 and all(re.match(r'(?:-?\d+-\d|beginning of time|end of time)', part) for part in dates):
        left, right = (time_text(x, 'day') for x in dates)
        end, start = (left, right) if re.search(r'\s+-\s+', text) else (right, left)
        if end == 'beginning of time' or start == 'end of time':
            raise ValueError('reversed open duration')
        if 'end of time' in (end, start) or 'beginning of time' in (end, start):
            return 'forever'
        days = (date.fromisoformat(end)-date.fromisoformat(start)).days
        if days < 0:
            raise ValueError('negative duration')
        return Decimal(days*86400)
    units = dict(year=365*86400, month=30*86400, week=7*86400, day=86400,
                 hour=3600, minute=60, second=1)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    pattern = r'(\d+(?:\.\d+)?)\s*(years?|months?|weeks?|days?|hours?|minutes?|seconds?)'
    parts = list(re.finditer(pattern, text))
    remainder = re.sub(pattern, '', text)
    if not parts or re.sub(r'\band\b', '', remainder).strip(' ,'):
        raise ValueError('expected an explicit duration with units')
    seconds = sum((Decimal(m[1])*units[m[2].rstrip('s')] for m in parts), Decimal(0))
    return seconds.normalize()


def readable_answer(value, kind, corpus):
    if kind == 'interval_set':
        return interval_answer(value, corpus)
    if isinstance(value, list):
        if not value:
            raise ValueError('no first answer')
        value = value[0]
    if isinstance(value, bool) or value is None:
        raise ValueError('expected an answer value')
    if kind in ('entity', 'entity_set'):
        return answer_name(value)
    if kind in ('year', 'month', 'day'):
        return time_text(value, kind)
    if kind == 'duration':
        return duration_seconds(value)
    if kind == 'rank':
        text = str(value).strip().lower()
        text = {'first': '1', 'second': '2', 'third': '3'}.get(text, text)
        match = re.fullmatch(r'([1-9]\d*)(?:st|nd|rd|th)?', text)
        if not match:
            raise ValueError('expected a positive rank')
        return int(match[1])
    if kind == 'comparison':
        text = normalize_answer(str(value)).rstrip('.')
        text = {'equal': 'equals', 'same': 'equals'}.get(text, text)
        if text not in ('shorter', 'longer', 'equals'):
            raise ValueError('expected shorter, longer or equals')
        return text
    raise ValueError(f'unsupported readable answer type: {kind}')


def duration_match(prediction, reference):
    value = prediction[0] if isinstance(prediction, list) and prediction else prediction
    text = str(value).casefold()
    raw = str(reference).strip().strip('()')
    spans = re.split(r'\s+-\s+|,\s*', raw)
    units = re.findall(r'(\d+(?:\.\d+)?)\s*(years?|months?)', text)
    if len(spans) == 2 and units:
        try:
            left, right = (date.fromisoformat(time_text(s, 'day')) for s in spans)
        except ValueError:
            pass
        else:
            end, start = (left, right) if re.search(r'\s+-\s+', raw) else (right, left)
            years = sum((Decimal(n) for n,u in units if u.startswith('year')), Decimal(0))
            months = sum((Decimal(n) for n,u in units if u.startswith('month')), Decimal(0))
            if years == int(years) and months == int(months):
                month_index = start.month-1+int(years)*12+int(months)
                year, month = start.year+month_index//12, month_index%12+1
                shifted = date(year, month, min(start.day, calendar.monthrange(year, month)[1]))
                seconds = duration_seconds(value)-(years*365+months*30)*86400
                return datetime.combine(shifted, datetime.min.time())+timedelta(seconds=float(seconds)) == datetime.combine(end, datetime.min.time())
    return readable_answer(prediction, 'duration', '') == readable_answer(reference, 'duration', '')


def empty_answer(payload, kind):
    if payload is None or "answers" not in payload:
        return False
    answers = payload["answers"]
    value = answers[0] if isinstance(answers, list) and len(answers) == 1 else answers
    text = str(value).strip().casefold()
    if answers is None or answers == [] or text in (
            "no answer", "no overlap", "no common period", "empty", "[]", "null", "none"):
        return True
    if kind == "duration" and not isinstance(value, (bool, list)):
        if text == "0":
            return True
        try:
            return duration_seconds(value) == 0
        except ValueError:
            pass
    return False


def match_answer(payload, task, corpus):
    kind, gold_values = task["answer_type"], task["answers"]
    first_only = task.get("score_mode") == "hits_at_one"
    if gold_values == [] and kind != "interval_set":
        correct = empty_answer(payload, kind)
        return dict(correct=correct, format_error=None, parsed_answer=[] if correct else None)
    set_kind = kind in ("entity", "entity_set", "day", "month", "year")
    gold = ([readable_answer(v, kind, corpus) for v in gold_values] if set_kind
            else [readable_answer(gold_values, kind, corpus)])
    if payload is None:
        return dict(correct=False, format_error="no_complete_answer", parsed_answer=None)
    try:
        submitted = payload.get("answers")
        if set_kind and not first_only:
            values = submitted if isinstance(submitted, list) else [submitted]
            parsed = sorted({readable_answer(v, kind, corpus) for v in values})
            correct = parsed == sorted(set(gold))
        else:
            parsed = readable_answer(submitted, kind, corpus)
            correct = (duration_match(submitted, gold_values[0]) if kind == "duration"
                       else parsed in gold)
    except (ValueError, TypeError) as error:
        return dict(correct=False, format_error=str(error), parsed_answer=None)
    return dict(correct=correct, format_error=None, parsed_answer=parsed)


def timelinekgqa_answer_kind(row, unit):
    kind = row['answer_type']
    if kind in ('subject', 'object'):
        return 'entity_set'
    if kind in ('timestamp_start', 'timestamp_end'):
        return unit
    if kind in ('timestamp_range', 'relation_union_or_intersection'):
        return 'interval_set'
    if kind == 'relation_ranking' or (row['temporal_relation'] == 'duration_compare' and row['question_level'] == 'complex'):
        return 'rank'
    if row['temporal_relation'] == 'duration_compare':
        return 'comparison'
    if kind in ('duration', 'relation_duration'):
        return 'duration'
    raise ValueError(f'Unknown TimelineKGQA answer type: {kind}')


def reference_task(corpus, row):
    if corpus == "MultiTQ":
        kind = "entity_set" if row["answer_type"] == "entity" else row["time_level"]
        raw = row["answers"]
    else:
        kind = timelinekgqa_answer_kind(row, "year" if corpus == "CronQuestion" else "day")
        raw = row["answer"]
    if corpus != "MultiTQ" and (raw is None or str(raw).strip().casefold() == "no answer"):
        values = []
    elif corpus == "MultiTQ":
        values = raw
    elif kind == "interval_set":
        values = interval_answer(raw, corpus)
    elif kind in ("day", "year"):
        values = [time_text(raw, kind)]
    else:
        values = [raw]
    return dict(answer_type=kind, answers=values,
                score_mode="exact" if kind == "interval_set" else "hits_at_one")


def point_interval_answer(corpus, value):
    if not isinstance(value, (str, int)):
        return value
    try:
        point = time_text(value, "year" if corpus == "CronQuestion" else "day")
    except (ValueError, TypeError):
        return value
    if point in ("beginning of time", "end of time"):
        return value
    return [point, point]


def score_prediction(corpus, row, prediction):
    if prediction["status"] == "infrastructure_error":
        raise RuntimeError("Infrastructure errors must be resolved before scoring")
    task = reference_task(corpus, row)
    payload = None
    if prediction["status"] == "complete":
        answer = prediction["answer"]
        if corpus != "MultiTQ" and task["answer_type"] == "interval_set":
            answer = point_interval_answer(corpus, answer)
        payload = {"answers": [answer]}
    return match_answer(payload, task, corpus)


def question_id(corpus, row):
    return str(row["quid"] if corpus == "MultiTQ" else row["id"])


def question_group(corpus, row):
    if corpus == "MultiTQ":
        return row["qtype"]
    kind = row["temporal_relation"] if row["answer_type"] == "relation_union_or_intersection" else row["answer_type"]
    return f"{row['question_level']}/{kind}"

