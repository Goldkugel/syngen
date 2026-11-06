"""
Builds the ConceptNet database of SynGenConceptNet.py (syngen.conceptnet.
database, SQLite) from the assertions dump (syngen.conceptnet.dump), with
only the relations and languages of syngen.conceptnet, so that the stage looks
words up in an index instead of reading the dump. Ported from
tools/build_conceptnet_db.py of the student project ba-syngen-conceptnet-llm.

The dump (ConceptNet 5.7, about 10 GB unpacked) is available at
https://github.com/commonsense/conceptnet5/wiki/Downloads, e.g.
conceptnet-assertions-5.7.0.csv.gz; it is read packed or unpacked.

    python3 ./src/SynGenConceptNetDB.py
"""

import gzip
import json
import os
import re
import shlex
import sqlite3
import sys
import time

sys.dont_write_bytecode = True

from SynGenUtils import *
from config      import *

from logger      import Logger

conceptnetConfig = syngenConfig.conceptnet

# /c/<language>/<word>[/<part of speech>[/<source>/<sense>]], e.g.
# /c/en/short_stature/n or /c/en/bank/n/wn/geography.
conceptPattern = re.compile(
    r"^/c/(?P<language>[a-z-]+)/(?P<word>[^/]+)"
    r"(?:/(?P<pos>[avrsn]))?(?:/[^/]+(?:/(?P<sense>[^/]+))?)?/?$")

schema = """
CREATE TABLE concepts (
    id       INTEGER PRIMARY KEY,
    word     TEXT NOT NULL,
    language TEXT NOT NULL,
    pos      TEXT NOT NULL,
    sense    TEXT NOT NULL
);
CREATE TABLE edges (
    source   INTEGER NOT NULL,
    target   INTEGER NOT NULL,
    relation TEXT NOT NULL,
    weight   REAL NOT NULL,
    PRIMARY KEY (source, target, relation)
) WITHOUT ROWID;
CREATE TABLE settings (
    settings TEXT NOT NULL
);
"""

# The settings the database is built with, recorded in it, so that
# SynGenConceptNet.py refuses a database built with others.
def databaseSettings() -> str:
    """The relations and languages of the database, as JSON."""
    return json.dumps({
        "relations" : sorted(conceptnetConfig.relations),
        "languages" : sorted(conceptnetConfig.languages)
    }, sort_keys = True)

def parseConcept(uri : str) -> tuple | None:
    """
    The (word, language, part of speech, sense) of a concept URI, the word in
    lower case with spaces, the part of speech "" and the sense "" if the URI
    has none; None for another URI.
    """
    match = conceptPattern.match(uri)
    if match is None:
        return None
    word = match.group("word").replace("_", " ").strip().lower()
    return word, match.group("language"), match.group("pos") or "", match.group("sense") or ""

def build(dump : str, database : str) -> None:
    """
    Reads the assertions of the configured relations between concepts of the
    configured languages from the dump into a new database (written next to
    it and renamed when complete).
    """
    l = Logger()
    exitIfFileNotExist(dump)
    folder = os.path.dirname(database)
    if folder:
        os.makedirs(folder, exist_ok = True)
    temporary = database + ".building"
    if os.path.exists(temporary):
        os.remove(temporary)

    connection = sqlite3.connect(temporary)
    connection.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
    connection.executescript(schema)

    relations = {f"/r/{relation}" : relation for relation in conceptnetConfig.relations}
    languages = set(conceptnetConfig.languages)
    concepts  = {}
    edges     = []
    kept      = 0
    started   = time.monotonic()

    def conceptID(uri : str) -> int | None:
        parsed = parseConcept(uri)
        if parsed is None or parsed[1] not in languages:
            return None
        if parsed not in concepts:
            concepts[parsed] = len(concepts) + 1
        return concepts[parsed]

    def flush() -> None:
        connection.executemany("INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)", edges)
        edges.clear()

    opener = gzip.open if dump.endswith(".gz") else open
    with opener(dump, "rt", encoding = "utf-8") as handle:
        for counter, line in enumerate(handle, start = 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or fields[1] not in relations:
                continue
            source, target = conceptID(fields[2]), conceptID(fields[3])
            if source is None or target is None or source == target:
                continue
            try:
                weight = float(json.loads(fields[4]).get("weight", 1.0))
            except (ValueError, AttributeError):
                weight = 1.0
            edges.append((source, target, relations[fields[1]], weight))
            kept += 1
            if len(edges) >= 100000:
                flush()
            if counter % 5000000 == 0:
                l.log(f"{counter // 1000000} million lines read, {len(concepts)} concepts, " \
                    f"{kept} edges, {time.monotonic() - started:.0f} s.")
    flush()

    connection.executemany("INSERT INTO concepts VALUES (?, ?, ?, ?, ?)",
        [(i, word, language, pos, sense) for (word, language, pos, sense), i in concepts.items()])
    connection.execute("INSERT INTO settings VALUES (?)", (databaseSettings(),))
    l.log("Building the indexes...")
    connection.execute("CREATE INDEX concepts_word ON concepts (word, language)")
    connection.execute("CREATE INDEX edges_target ON edges (target, relation)")
    connection.execute("ANALYZE")
    connection.commit()
    connection.close()
    os.replace(temporary, database)
    l.log(f"{len(concepts)} concepts and {kept} edges of the relations " \
        f"{applyFormat(sorted(relations.values()))} and languages {applyFormat(sorted(languages))}.")


if __name__ == "__main__":
    l = Logger(config = configFile)
    l.log(f"Command: python3 {shlex.join(sys.argv)}")
    l.printHeader("Building the ConceptNet Database")
    with l.step("Building the ConceptNet database"):
        build(conceptnetConfig.dump, conceptnetConfig.database)
