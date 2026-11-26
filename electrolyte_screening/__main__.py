"""Command line interface for downloading, auditing, training, and prediction."""

import argparse
from pathlib import Path

from .audit import audit
from .data import download
from .features import CifInputError
from .models import benchmark, predict_cif, write_report
from .verification import verify_uploads

DEFAULT_DATA = Path("data/raw")
DEFAULT_MODEL = Path("artifacts/fast_conductor.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast lithium-conductor screening with OBELiX.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download and verify the pinned official files.")
    inspect = commands.add_parser("audit", help="Summarize the official splits and CIF coverage.")
    train = commands.add_parser("train", help="Select, evaluate, and save a model.")
    predict = commands.add_parser("predict", help="Score a lithium-containing CIF.")
    verify = commands.add_parser("verify-uploads", help="Verify archived test CIFs and composition overlap.")
    for command in (fetch, inspect, train, verify):
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    for command in (train, predict, verify):
        command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    for command in (inspect, train, verify):
        command.add_argument("--report", type=Path)
    predict.add_argument("cif", type=Path)
    args = parser.parse_args()

    if args.command == "download":
        download(args.data_dir)
    elif args.command == "audit":
        write_report(args.report, audit(args.data_dir))
    elif args.command == "train":
        write_report(args.report, benchmark(args.data_dir, args.model))
    elif args.command == "verify-uploads":
        write_report(args.report, verify_uploads(args.data_dir, args.model))
    else:
        try:
            write_report(None, predict_cif(args.model, args.cif))
        except CifInputError as error:
            predict.error(str(error))


if __name__ == "__main__":
    main()
