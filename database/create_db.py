"""Build the DELTA database.

    python create_db.py                  # new, empty delta.db
    python create_db.py --seed           # ...with the sample data
    python create_db.py --reset --seed   # delete the old one and start over

Refuses to touch an existing delta.db unless you pass --reset, because
rebuilding deletes every row in it.
"""

import argparse
import sys

import db
import seed


def main():
    parser = argparse.ArgumentParser(description="Create the DELTA database.")
    parser.add_argument("--reset", action="store_true",
                        help="delete an existing database and rebuild it")
    parser.add_argument("--seed", action="store_true",
                        help="fill it with sample data")
    args = parser.parse_args()

    try:
        conn = db.init_db(reset=args.reset)
    except FileExistsError as error:
        print(f"Not changed: {error}", file=sys.stderr)
        return 1

    if args.seed:
        seed.seed(conn)

    print(f"Created {db.db_path()}")

    for table in db.TABLES:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<13} {count} rows")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
