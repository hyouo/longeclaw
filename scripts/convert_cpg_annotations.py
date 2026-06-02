"""
Convert CpG annotation JSON (from seqprompts pipeline) to SQLite database
for use by LongevityClaw's CpG annotation lookup tool.

Source: cpg_annotations.json (73 MB, ~31.5K CpGs, GRCh38)
Output: data/cpg_annotations.sqlite (~40 MB)

Tables:
  cpg_main    — one row per CpG: coordinates, sequence, island context, ENCODE cCRE, gene density
  cpg_genes   — CpG ↔ gene relations: region (TSS200, Body, etc.), biotype, TSS distance
  cpg_nearest — ~10 nearest genes per CpG by TSS distance (regardless of overlap)
  cpg_tfbs    — transcription factor binding sites overlapping each CpG
  _metadata   — key/value provenance info

Usage:
    python scripts/convert_cpg_annotations.py [path/to/cpg_annotations.json]

Default input: C:/Users/LENOVO/Desktop/ClaudeCode/seqprompts/cpg_annotation_output/cpg_annotations.json
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_INPUT = Path(r"C:\Users\LENOVO\Desktop\ClaudeCode\seqprompts\cpg_annotation_output\cpg_annotations.json")
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "cpg_annotations.sqlite"


def parse_genome_address(addr: str) -> tuple[str, int, str]:
    m = re.match(r"(chr\w+):(\d+),\s*(minus|plus)\s+strand", addr)
    if not m:
        return ("", 0, "")
    return m.group(1), int(m.group(2)), "-" if m.group(3) == "minus" else "+"


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    print(f"Reading {input_path} ...")
    with open(input_path) as f:
        entries = json.load(f)
    print(f"  {len(entries)} CpG entries")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    conn = sqlite3.connect(str(OUTPUT))
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE cpg_main (
            cg_id            TEXT PRIMARY KEY,
            chr              TEXT NOT NULL,
            position         INTEGER NOT NULL,
            strand           TEXT,
            genome_seq       TEXT,
            gene_density     INTEGER,
            island_relation  TEXT,
            island_id        TEXT,
            island_start     INTEGER,
            island_end       INTEGER,
            dist_to_island   INTEGER,
            encode_type      TEXT,
            encode_ctcf      INTEGER,
            encode_element   TEXT
        );

        CREATE TABLE cpg_genes (
            cg_id       TEXT NOT NULL,
            gene        TEXT NOT NULL,
            biotype     TEXT,
            region      TEXT,
            subregion   TEXT,
            tss_dist    INTEGER,
            FOREIGN KEY (cg_id) REFERENCES cpg_main(cg_id)
        );

        CREATE TABLE cpg_nearest (
            cg_id       TEXT NOT NULL,
            gene        TEXT NOT NULL,
            biotype     TEXT,
            tss_dist    INTEGER,
            FOREIGN KEY (cg_id) REFERENCES cpg_main(cg_id)
        );

        CREATE TABLE cpg_tfbs (
            cg_id       TEXT NOT NULL,
            tf          TEXT NOT NULL,
            score_pct   INTEGER,
            offset      INTEGER,
            strand      TEXT,
            matched_seq TEXT,
            FOREIGN KEY (cg_id) REFERENCES cpg_main(cg_id)
        );

        CREATE TABLE _metadata (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)

    main_rows = []
    gene_rows = []
    nearest_rows = []
    tfbs_rows = []

    for entry in entries:
        cg_id = entry["cg_id"]
        chrom, pos, strand = parse_genome_address(entry.get("genome_address", ""))

        island = entry.get("cpg_island_context") or {}
        encode = entry.get("encode_ccre") or {}

        main_rows.append((
            cg_id, chrom, pos, strand,
            entry.get("genome_seq"),
            entry.get("gene_density_500kbp"),
            island.get("relation"),
            island.get("island_id"),
            island.get("island_start"),
            island.get("island_end"),
            island.get("dist_to_island"),
            encode.get("type"),
            1 if encode.get("ctcf_bound") else (0 if encode else None),
            encode.get("element_id"),
        ))

        for gene, info in (entry.get("gene_relations") or {}).items():
            gene_rows.append((
                cg_id, gene,
                info.get("biotype"),
                info.get("region"),
                info.get("subregion"),
                info.get("tss_dist_directed_nt"),
            ))

        for ng in (entry.get("nearest_genes") or []):
            nearest_rows.append((
                cg_id, ng["gene"],
                ng.get("biotype"),
                ng.get("tss_dist_nt"),
            ))

        for tfbs in (entry.get("tfbs") or []):
            tfbs_rows.append((
                cg_id, tfbs["tf"],
                tfbs.get("score_pct"),
                tfbs.get("offset_from_cpg"),
                tfbs.get("strand"),
                tfbs.get("matched_seq"),
            ))

    print(f"Inserting {len(main_rows)} CpGs, {len(gene_rows)} gene relations, "
          f"{len(nearest_rows)} nearest genes, {len(tfbs_rows)} TFBS ...")

    cur.executemany("INSERT INTO cpg_main VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", main_rows)
    cur.executemany("INSERT INTO cpg_genes VALUES (?,?,?,?,?,?)", gene_rows)
    cur.executemany("INSERT INTO cpg_nearest VALUES (?,?,?,?)", nearest_rows)
    cur.executemany("INSERT INTO cpg_tfbs VALUES (?,?,?,?,?,?)", tfbs_rows)

    cur.executemany("INSERT INTO _metadata VALUES (?,?)", [
        ("genome_build", "GRCh38"),
        ("source_date", "2026-02-27"),
        ("sources", "450K/EPIC manifests, Ensembl GRCh38, UCSC CpG Islands, ENCODE cCREs, JASPAR2026"),
        ("n_cpgs", str(len(main_rows))),
        ("n_gene_relations", str(len(gene_rows))),
        ("n_nearest_genes", str(len(nearest_rows))),
        ("n_tfbs", str(len(tfbs_rows))),
    ])

    print("Creating indexes ...")
    cur.executescript("""
        CREATE INDEX idx_cpg_main_chr ON cpg_main(chr);
        CREATE INDEX idx_cpg_genes_gene ON cpg_genes(gene);
        CREATE INDEX idx_cpg_genes_region ON cpg_genes(region);
        CREATE INDEX idx_cpg_genes_cg ON cpg_genes(cg_id);
        CREATE INDEX idx_cpg_nearest_cg ON cpg_nearest(cg_id);
        CREATE INDEX idx_cpg_nearest_gene ON cpg_nearest(gene);
        CREATE INDEX idx_cpg_tfbs_cg ON cpg_tfbs(cg_id);
        CREATE INDEX idx_cpg_tfbs_tf ON cpg_tfbs(tf);
    """)

    conn.commit()
    conn.close()

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Done: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
