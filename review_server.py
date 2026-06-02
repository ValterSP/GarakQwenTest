import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def safe_file_part(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return cleaned or "unknown"


def review_file_name(model, probe):
    return f"{safe_file_part(model)}_{safe_file_part(probe)}_misclassified_reviews.jsonl"


def read_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            review_key = row.get("review_key")
            if review_key:
                rows[str(review_key)] = row
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(rows.values())
    items.sort(key=lambda row: str(row.get("reviewed_at") or ""))
    with path.open("w", encoding="utf-8") as handle:
        for row in items:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "GarakReviewServer/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reviews":
            self.handle_get_reviews(parsed)
            return
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reviews":
            self.handle_post_review()
            return
        self.send_json({"error": "Not found"}, status=404)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def handle_get_reviews(self, parsed):
        query = parse_qs(parsed.query)
        model = (query.get("model") or [""])[0]
        probe = (query.get("probe") or [""])[0]
        if not model or not probe:
            self.send_json({"error": "Missing model or probe"}, status=400)
            return

        path = self.server.review_dir / review_file_name(model, probe)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        rows = read_jsonl(path)
        self.send_json({"rows": rows, "count": len(rows), "file": str(path)})

    def handle_post_review(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            record = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON body"}, status=400)
            return

        required = ["review_key", "model", "probe"]
        missing = [key for key in required if not record.get(key)]
        if missing:
            self.send_json({"error": f"Missing field(s): {', '.join(missing)}"}, status=400)
            return

        path = self.server.review_dir / review_file_name(record["model"], record["probe"])
        rows = read_jsonl(path)
        already_exists = str(record["review_key"]) in rows
        if not already_exists:
            rows[str(record["review_key"])] = record
            write_jsonl(path, rows)

        self.send_json(
            {
                "rows": rows,
                "count": len(rows),
                "file": str(path),
                "already_exists": already_exists,
            }
        )

    def serve_static(self, request_path):
        if request_path in ("", "/"):
            request_path = "/" + self.server.index.name
        relative = request_path.lstrip("/")
        path = (self.server.root / relative).resolve()

        if not path.is_file() or not path.is_relative_to(self.server.root):
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Serve report_chat_browser.html with JSONL review write APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--index", type=Path, default=Path("report_chat_browser.html"))
    parser.add_argument("--review-dir", type=Path, default=Path("garakManualReviews"))
    args = parser.parse_args()

    root = args.root.resolve()
    index = (root / args.index).resolve()
    review_dir = (root / args.review_dir).resolve()

    if not index.is_file():
        raise FileNotFoundError(index)

    httpd = ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler)
    httpd.root = root
    httpd.index = index
    httpd.review_dir = review_dir

    print(f"Serving {index.name} at http://{args.host}:{args.port}/")
    print(f"Writing manual reviews to {review_dir}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
