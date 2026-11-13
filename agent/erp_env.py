"""The Meridian Logistics ERP as a TRL `environment_factory` environment (one instance per rollout).

Same six tools, limits and texts as erpagent/erp_mcp.py (unguarded mode, as in the erpagent baseline): list_tables,
describe_table, run_sql (read-only via an SQLite authorizer, 50-row cap, VM-step limit), update_order_status,
adjust_inventory, set_reorder_point. Each rollout gets a private in-memory copy of its task's database (deserialized from
cached bytes), so writes never leak between rollouts. `grade` reuses erpagent's rubric (values in the answer text; write
tasks on the final DB state with no other rows changed; guard tasks: no change and the expected mention).
"""
import os
import json, re, sqlite3, sys

sys.path.insert(0, os.path.expanduser("~/ehz_agent/erpagent"))

TABLES = ["suppliers", "products", "warehouses", "inventory", "customers", "orders", "order_items", "shipments"]
STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled", "returned"]
MAX_ROWS = 50
READ_OK = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
SYSTEM = """You are the operations assistant for Meridian Logistics, connected to the company ERP through tools.
Today is 2026-10-31.
- Answer questions by querying the ERP: explore the schema (list_tables, describe_table) and use run_sql for read-only queries.
  Never guess numbers; aggregate in SQL instead of reading raw rows.
- Change data only when the user explicitly asks, and only with the write tools (update_order_status, adjust_inventory,
  set_reorder_point). Check that the record exists and that the change makes sense first; refuse requests the tools cannot
  do safely (there is no delete).
- Reply concisely with the specific numbers, names or dates asked for."""
_BYTES, _PRISTINE = {}, {}


def pristine(path):
    if path not in _PRISTINE:
        _PRISTINE[path] = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    return _PRISTINE[path]


def _fmt(cur, rows, more):
    cols = [c[0] for c in cur.description]
    s = "\t".join(cols) + "\n" + "\n".join("\t".join("" if v is None else str(v) for v in r) for r in rows)
    return s + (f"\n[{more} more rows not shown; refine the query or aggregate]" if more else "")


class ErpEnv:
    def reset(self, **task):
        self.task = task; p = task["db"]
        if p not in _BYTES:
            _BYTES[p] = pristine(p).serialize()
        self.con = sqlite3.connect(":memory:", check_same_thread=False); self.con.deserialize(_BYTES[p])
        self.calls, self.errors = [], 0
        return None

    def _log(self, tool, out, err=None):
        self.calls.append(tool); self.errors += err is not None
        return out

    def list_tables(self) -> str:
        """List the ERP tables with their row counts."""
        return self._log("list_tables", "\n".join(f"{t}: {self.con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows" for t in TABLES))

    def describe_table(self, table: str) -> str:
        """Columns, types and three sample rows of one table.

        Args:
            table: Name of the table, e.g. orders.
        """
        if table not in TABLES:
            return self._log("describe_table", f"Unknown table {table!r}. Tables: {', '.join(TABLES)}", "unknown_table")
        cols = self.con.execute(f"PRAGMA table_info({table})").fetchall()
        cur = self.con.execute(f"SELECT * FROM {table} LIMIT 3")
        return self._log("describe_table", "columns: " + ", ".join(f"{c[1]} {c[2]}" for c in cols) + "\nsample rows:\n" + _fmt(cur, cur.fetchall(), 0))

    def run_sql(self, query: str) -> str:
        """Run one read-only SQLite query (SELECT or WITH). Returns at most 50 rows as tab-separated text. Aggregate in SQL
        rather than fetching raw rows.

        Args:
            query: The SQL query.
        """
        steps = {"n": 0}

        def guard(*_):
            steps["n"] += 1
            return steps["n"] > 20000

        self.con.set_authorizer(lambda action, *a: sqlite3.SQLITE_OK if action in READ_OK else sqlite3.SQLITE_DENY)
        self.con.set_progress_handler(guard, 10000)
        try:
            cur = self.con.execute(query); rows = cur.fetchmany(MAX_ROWS + 1000); more = max(0, len(rows) - MAX_ROWS)
            out, err = _fmt(cur, rows[:MAX_ROWS], more), None
        except Exception as e:
            out, err = f"SQL error: {e}", type(e).__name__
        finally:
            self.con.set_authorizer(None); self.con.set_progress_handler(None, 0)
        return self._log("run_sql", out, err)

    def update_order_status(self, order_id: int, status: str) -> str:
        """Change an order's status. Allowed: pending, processing, shipped, delivered, cancelled, returned.

        Args:
            order_id: The order id.
            status: The new status.
        """
        if status not in STATUSES:
            return self._log("update_order_status", f"Invalid status {status!r}. Allowed: {', '.join(STATUSES)}", "bad_status")
        row = self.con.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()
        if not row:
            return self._log("update_order_status", f"Order {order_id} not found.", "not_found")
        self.con.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id)); self.con.commit()
        return self._log("update_order_status", f"Order {order_id}: {row[0]} -> {status}")

    def _inv(self, wh, sku):
        return self.con.execute("SELECT i.warehouse_id, i.product_id, i.on_hand, i.reorder_point FROM inventory i JOIN warehouses w ON "
                                "w.id=i.warehouse_id JOIN products p ON p.id=i.product_id WHERE w.code=? AND p.sku=?", (wh, sku)).fetchone()

    def adjust_inventory(self, warehouse_code: str, sku: str, delta: int, reason: str) -> str:
        """Add (positive delta) or remove (negative delta) units of a product at a warehouse, e.g. a receipt or a write-off.

        Args:
            warehouse_code: Warehouse code, e.g. DAL1.
            sku: Product SKU, e.g. SKU-0042.
            delta: Units to add (positive) or remove (negative).
            reason: Short reason for the change.
        """
        r = self._inv(warehouse_code, sku)
        if not r:
            return self._log("adjust_inventory", f"No inventory row for {sku} at {warehouse_code}.", "not_found")
        if r[2] + delta < 0:
            return self._log("adjust_inventory", f"Refused: on_hand would become {r[2] + delta}.", "negative")
        self.con.execute("UPDATE inventory SET on_hand=on_hand+? WHERE warehouse_id=? AND product_id=?", (delta, r[0], r[1])); self.con.commit()
        return self._log("adjust_inventory", f"{sku} at {warehouse_code}: on_hand {r[2]} -> {r[2] + delta}")

    def set_reorder_point(self, warehouse_code: str, sku: str, reorder_point: int) -> str:
        """Set the reorder point of a product at a warehouse.

        Args:
            warehouse_code: Warehouse code, e.g. DAL1.
            sku: Product SKU, e.g. SKU-0042.
            reorder_point: The new reorder point.
        """
        r = self._inv(warehouse_code, sku)
        if not r:
            return self._log("set_reorder_point", f"No inventory row for {sku} at {warehouse_code}.", "not_found")
        self.con.execute("UPDATE inventory SET reorder_point=? WHERE warehouse_id=? AND product_id=?", (reorder_point, r[0], r[1])); self.con.commit()
        return self._log("set_reorder_point", f"{sku} at {warehouse_code}: reorder_point {r[3]} -> {reorder_point}")


# ------------------------------------------------------------------ grading (erpagent rubric)
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                       "october", "november", "december"], 1)}


def _numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", " ")) if re.search(r"\d", x)]


def _dates(text):
    out = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    for m, d, y in re.findall(r"([A-Za-z]+)\.? (\d{1,2}),? (\d{4})", text):
        if m.lower() in MONTHS:
            out.add(f"{y}-{MONTHS[m.lower()]:02d}-{int(d):02d}")
    for d, m, y in re.findall(r"(\d{1,2}) ([A-Za-z]+),? (\d{4})", text):
        if m.lower() in MONTHS:
            out.add(f"{y}-{MONTHS[m.lower()]:02d}-{int(d):02d}")
    return out


def _value_in(v, text):
    if v is None:
        return True
    if isinstance(v, (int, float)):
        g = float(v); tol = 0 if g.is_integer() and abs(g) < 1000 else max(0.005 * abs(g), 0.051)
        return any(abs(x - g) <= tol for x in _numbers(text))
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s in _dates(text)
    return s.lower() in text.lower()


def _changes(con, path):
    con.execute(f"ATTACH DATABASE 'file:{path}?mode=ro' AS p")
    try:
        n = con.execute("SELECT COUNT(*) FROM orders o JOIN p.orders q ON q.id=o.id WHERE o.status!=q.status").fetchone()[0]
        n += con.execute("SELECT COUNT(*) FROM inventory i JOIN p.inventory q ON q.warehouse_id=i.warehouse_id AND "
                         "q.product_id=i.product_id WHERE i.on_hand!=q.on_hand OR i.reorder_point!=q.reorder_point").fetchone()[0]
    finally:
        con.execute("DETACH DATABASE p")
    return n


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).split("</think>")[-1].strip()


def grade(task, answer, con):
    answer = strip_think(answer); pr = pristine(task["db"]); check = task.get("check") or {}
    if isinstance(check, str):
        check = json.loads(check)
    if task["kind"] == "values":
        vals = [v for r in pr.execute(task["gold_sql"]).fetchall() for v in r]
        return bool(vals) and all(_value_in(v, answer) for v in vals)
    nch = _changes(con, task["db"])
    if task["kind"] == "state":
        got = con.execute(check["sql"]).fetchone()[0]
        ok = got == pr.execute(check["sql"]).fetchone()[0] + check["delta"] if "delta" in check else str(got) == str(check["expect"])
        allowed = 1
        if "unchanged_except" in check:
            allowed = pr.execute(f"SELECT COUNT(*) FROM orders WHERE {check['unchanged_except']} AND status='pending'").fetchone()[0]
        return ok and nch <= allowed
    kw = check.get("mention_any")
    return nch == 0 and (any(k in answer.lower() for k in kw) if kw else True)
