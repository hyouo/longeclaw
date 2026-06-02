"""
CpG genomic annotation database.
Lazy singleton wrapping a SQLite database of ~31K CpG annotations (GRCh38).
Provides forward lookup (CpG → annotations) and reverse search (gene/chr/region → CpGs).

Database built by scripts/convert_cpg_annotations.py from the seqprompts annotation pipeline.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cpg_annotations.sqlite"

_instance: "CpGDatabase | None" = None


def get_cpg_db() -> "CpGDatabase":
    global _instance
    if _instance is None:
        _instance = CpGDatabase()
    return _instance


class CpGDatabase:
    def __init__(self, db_path: Path | str = DB_PATH):
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"CpG annotation database not found: {self._db_path}\n"
                    "Run: python scripts/convert_cpg_annotations.py"
                )
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get_metadata(self) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT key, value FROM _metadata").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def lookup(self, cg_ids: list[str]) -> list[dict]:
        conn = self._get_conn()
        results = []
        for cg_id in cg_ids:
            row = conn.execute("SELECT * FROM cpg_main WHERE cg_id = ?", (cg_id,)).fetchone()
            if not row:
                results.append({"cg_id": cg_id, "found": False})
                continue
            results.append(self._enrich_main_row(conn, dict(row), include_tfbs=True))
        return results

    def search(
        self,
        chromosome: str | None = None,
        gene: str | None = None,
        region: str | None = None,
        island_relation: str | None = None,
        encode_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conn = self._get_conn()

        if gene or region:
            return self._search_via_genes(conn, chromosome, gene, region, island_relation, encode_type, limit)

        conditions = []
        params: list = []
        if chromosome:
            conditions.append("chr = ?")
            params.append(chromosome)
        if island_relation:
            conditions.append("island_relation = ?")
            params.append(island_relation)
        if encode_type:
            conditions.append("encode_type = ?")
            params.append(encode_type)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM cpg_main WHERE {where} LIMIT ?",
            params + [limit],
        ).fetchall()

        return [self._enrich_main_row(conn, dict(r)) for r in rows]

    def _search_via_genes(
        self,
        conn: sqlite3.Connection,
        chromosome: str | None,
        gene: str | None,
        region: str | None,
        island_relation: str | None,
        encode_type: str | None,
        limit: int,
    ) -> list[dict]:
        gene_conds = []
        params: list = []
        if gene:
            gene_conds.append("UPPER(g.gene) = ?")
            params.append(gene.upper())
        if region:
            gene_conds.append("g.region = ?")
            params.append(region)

        main_conds = []
        if chromosome:
            main_conds.append("m.chr = ?")
            params.append(chromosome)
        if island_relation:
            main_conds.append("m.island_relation = ?")
            params.append(island_relation)
        if encode_type:
            main_conds.append("m.encode_type = ?")
            params.append(encode_type)

        gene_where = " AND ".join(gene_conds) if gene_conds else "1=1"
        main_where = " AND ".join(main_conds) if main_conds else "1=1"

        query = f"""
            SELECT DISTINCT m.* FROM cpg_main m
            JOIN cpg_genes g ON g.cg_id = m.cg_id
            WHERE {gene_where} AND {main_where}
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._enrich_main_row(conn, dict(r)) for r in rows]

    def _enrich_main_row(self, conn: sqlite3.Connection, entry: dict, include_tfbs: bool = False) -> dict:
        cg_id = entry["cg_id"]
        entry["found"] = True
        entry["encode_ctcf"] = bool(entry["encode_ctcf"]) if entry["encode_ctcf"] is not None else None

        genes = conn.execute(
            "SELECT gene, biotype, region, subregion, tss_dist FROM cpg_genes WHERE cg_id = ?",
            (cg_id,),
        ).fetchall()
        entry["genes"] = [dict(g) for g in genes]

        nearest = conn.execute(
            "SELECT gene, biotype, tss_dist FROM cpg_nearest WHERE cg_id = ? ORDER BY ABS(tss_dist)",
            (cg_id,),
        ).fetchall()
        entry["nearest_genes"] = [dict(n) for n in nearest]

        if include_tfbs:
            tfbs = conn.execute(
                "SELECT tf, score_pct, offset, strand, matched_seq FROM cpg_tfbs WHERE cg_id = ?",
                (cg_id,),
            ).fetchall()
            entry["tfbs"] = [dict(t) for t in tfbs]

        return entry
