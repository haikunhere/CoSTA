import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .scoring import question_group, question_id, score_prediction


def evaluate(corpus, questions, predictions, output):
    rows = json.loads(questions.read_text())
    by_id = {question_id(corpus, row): row for row in rows}
    fields = ("question_id", "status", "answer", "final_notice", "workers", "generated_tokens", "calls")
    read = gzip.open if predictions.suffix == ".gz" else open
    with read(predictions, "rt", encoding="utf-8") as stream:
        saved = [
            {key: row[key] for key in fields if key in row}
            for row in map(json.loads, (line for line in stream if line.strip()))
        ]
    if len(saved) != len(rows) or len({item["question_id"] for item in saved}) != len(rows):
        raise ValueError("Require exactly one complete prediction per question")
    if {item["question_id"] for item in saved} != set(by_id):
        raise ValueError("Prediction IDs differ from the supplied question file")
    output.mkdir(parents=True, exist_ok=True)
    totals, groups, statuses = Counter(), defaultdict(Counter), Counter()
    with (output / "scores.jsonl").open("w") as stream:
        for prediction in saved:
            row = by_id[prediction["question_id"]]
            score = score_prediction(corpus, row, prediction)
            group = question_group(corpus, row)
            result = dict(question_id=prediction["question_id"], group=group, **score)
            stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            for cell in (totals, groups[group]):
                cell["total"] += 1
                cell["correct"] += score["correct"] is True
                cell["prediction_parse_errors"] += score["format_error"] is not None
            statuses[prediction["status"]] += 1
    summary = dict(status="completed", corpus=corpus, total=totals["total"], correct=totals["correct"],
                   hits_at_one=totals["correct"] / totals["total"],
                   final_notice=sum(p.get("final_notice", False) for p in saved),
                   mean_workers=sum(p["workers"] for p in saved) / len(saved),
                   mean_generated_tokens=sum(p["generated_tokens"] for p in saved) / len(saved),
                   mean_calls=sum(p["calls"] for p in saved) / len(saved),
                   statuses=dict(statuses), groups=dict(groups),
                   scoring_rule="readable_v2; first answer, exact interval sets",
                   questions=str(questions.resolve()), predictions=str(predictions.resolve()))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--corpus", choices=["MultiTQ", "CronQuestion", "ICEWS"], required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.corpus, args.questions, args.predictions, args.output)


if __name__ == "__main__":
    main()
