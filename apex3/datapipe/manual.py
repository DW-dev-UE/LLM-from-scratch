from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from html import unescape
from typing import Iterable, Iterator

from .io import ShardWriter

SE_SITES = ["stackoverflow.com", "math.stackexchange.com", "mathoverflow.net", "stats.stackexchange.com", "physics.stackexchange.com", "cs.stackexchange.com", "cstheory.stackexchange.com", "datascience.stackexchange.com", "ai.stackexchange.com", "unix.stackexchange.com", "askubuntu.com", "superuser.com", "serverfault.com", "softwareengineering.stackexchange.com", "codereview.stackexchange.com", "dba.stackexchange.com", "tex.stackexchange.com", "security.stackexchange.com", "electronics.stackexchange.com",
            "chemistry.stackexchange.com", "biology.stackexchange.com", "bioinformatics.stackexchange.com", "quant.stackexchange.com", "scicomp.stackexchange.com", "crypto.stackexchange.com", "gis.stackexchange.com", "gamedev.stackexchange.com", "networkengineering.stackexchange.com", "devops.stackexchange.com", "mathematica.stackexchange.com", "economics.stackexchange.com", "engineering.stackexchange.com", "astronomy.stackexchange.com", "earthscience.stackexchange.com", "matheducators.stackexchange.com"]
SE_SITES_EXTRA = [f"{s}.stackexchange.com" for s in ("android", "arduino", "raspberrypi", "blender", "wordpress", "salesforce", "sharepoint", "magento", "drupal", "reverseengineering", "robotics", "sqa", "webmasters", "apple", "emacs", "vi", "retrocomputing", "quantumcomputing", "iot", "ethereum", "dsp", "computergraphics", "or", "opendata")]
SE_URL = "https://archive.org/download/stackexchange/{file}"


def se_archive_file(site: str) -> str:
    return "stackoverflow.com-Posts.7z" if site == "stackoverflow.com" else f"{site}.7z"


def html_to_text(html: str) -> str:
    s = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", lambda m: f"\n```\n{m.group(1).strip()}\n```\n", html, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<li[^>]*>", "- ", s)
    s = re.sub(r"</(p|div|h[1-6]|blockquote|ul|ol|table)>", "\n\n", s)
    s = re.sub(r"</(li|tr)>", "\n", s)
    s = re.sub(r"<h[1-6][^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = unescape(s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def se_rows(xml_stream) -> Iterator[str]:
    context = ET.iterparse(xml_stream, events=("start", "end"))
    _, root = next(context)
    for event, el in context:
        if event != "end" or el.tag != "row":
            continue
        a = el.attrib
        kind = a.get("PostTypeId")
        if kind == "1" and not a.get("ClosedDate"):
            rec = {"id": a["Id"], "title": a.get("Title", ""), "body": a.get("Body", ""), "score": int(a.get("Score", 0)), "accepted": a.get("AcceptedAnswerId"), "tags": a.get("Tags", "")}
            yield f"{a['Id']}\t0\t{json.dumps(rec, ensure_ascii=False)}\n"
        elif kind == "2" and a.get("ParentId"):
            rec = {"id": a["Id"], "body": a.get("Body", ""), "score": int(a.get("Score", 0))}
            yield f"{a['ParentId']}\t1\t{json.dumps(rec, ensure_ascii=False)}\n"
        root.clear()


def se_tags(tags: str) -> list[str]:
    return [t for t in re.split(r"[<>|]", tags) if t]


def se_doc(site: str, q: dict, answers: list[dict], max_answers: int = 3, min_score: int = 1) -> dict | None:
    if q["score"] < 0:
        return None
    accepted = [a for a in answers if a["id"] == q.get("accepted")]
    others = sorted((a for a in answers if a["id"] != q.get("accepted") and a["score"] >= min_score), key=lambda a: -a["score"])
    chosen = (accepted + others)[:max_answers]
    if not chosen:
        return None
    parts = [q["title"], html_to_text(q["body"])]
    tags = se_tags(q.get("tags", ""))
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    for a in chosen:
        parts.append("Answer:\n" + html_to_text(a["body"]))
    return {"id": f"{site}/{q['id']}", "text": "\n\n".join(parts), "url": f"https://{site}/questions/{q['id']}", "site": site, "score": q["score"], "license": "CC BY-SA 4.0"}


def se_join(sorted_lines: Iterable[str], site: str) -> Iterator[dict]:
    cur, q, answers = None, None, []
    for line in sorted_lines:
        key, kind, rec = line.rstrip("\n").split("\t", 2)
        if key != cur:
            if q is not None:
                doc = se_doc(site, q, answers)
                if doc:
                    yield doc
            cur, q, answers = key, None, []
        if kind == "0":
            q = json.loads(rec)
        else:
            answers.append(json.loads(rec))
    if q is not None:
        doc = se_doc(site, q, answers)
        if doc:
            yield doc


def fetch_stackexchange(site: str, out_dir: str, work: str, sort_mem: str = "8G", threads: int = 8) -> dict:
    os.makedirs(work, exist_ok=True)
    archive = os.path.join(work, se_archive_file(site))
    subprocess.run(["curl", "-sfL", "--retry", "5", "-o", archive, SE_URL.format(file=se_archive_file(site))], check=True)
    rows, sorted_rows = os.path.join(work, f"{site}.rows"), os.path.join(work, f"{site}.sorted")
    seven = subprocess.Popen(["7z", "e", "-so", archive, "Posts.xml"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with open(rows, "w", encoding="utf-8") as f:
        f.writelines(se_rows(seven.stdout))
    if seven.wait() != 0:
        raise RuntimeError(f"7z failed on {archive}")
    os.remove(archive)
    subprocess.run(["sort", "-t", "\t", "-k1,1n", "-k2,2n", "-S", sort_mem, f"--parallel={threads}", "-T", work, "-o", sorted_rows, rows], check=True, env={**os.environ, "LC_ALL": "C"})
    os.remove(rows)
    docs = 0
    with open(sorted_rows, encoding="utf-8") as f, ShardWriter(out_dir, site.replace(".", "_")) as w:
        for doc in se_join(f, site):
            w.write(doc)
            docs += 1
    os.remove(sorted_rows)
    return {"site": site, "docs": docs}


PUBMED_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


def pubmed_files() -> list[str]:
    import urllib.request

    with urllib.request.urlopen(PUBMED_BASE, timeout=60) as r:
        index = r.read().decode()
    return sorted(set(re.findall(r'(pubmed\d+n\d+\.xml\.gz)"', index)))


def pubmed_docs(stream) -> Iterator[dict]:
    context = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    for event, el in context:
        if event != "end" or el.tag != "PubmedArticle":
            continue
        cit = el.find("MedlineCitation")
        art = cit.find("Article") if cit is not None else None
        if art is not None and art.findtext("Language", "eng") == "eng":
            title_el = art.find("ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            parts = []
            for ab in art.findall("Abstract/AbstractText"):
                text = " ".join("".join(ab.itertext()).split())
                if text:
                    label = ab.get("Label")
                    parts.append(f"{label.capitalize()}: {text}" if label and label.upper() != "UNLABELLED" else text)
            if parts:
                pmid = cit.findtext("PMID")
                yield {"id": f"pmid{pmid}", "text": "\n\n".join([title] + parts) if title else "\n\n".join(parts), "pmid": pmid}
        root.clear()


def fetch_pubmed_file(name: str, out_dir: str, attempts: int = 4) -> dict:
    import glob
    import time
    import urllib.request

    prefix = name.split(".")[0]
    done = os.path.join(out_dir, ".done", prefix)
    if os.path.exists(done):
        return {"file": name, "skipped": True}
    error = None
    for attempt in range(attempts):
        for old in glob.glob(os.path.join(out_dir, f"{prefix}_*.jsonl.zst")):
            os.remove(old)
        try:
            docs = 0
            with urllib.request.urlopen(PUBMED_BASE + name, timeout=300) as r:
                with gzip.GzipFile(fileobj=r) as f, ShardWriter(out_dir, prefix) as w:
                    for doc in pubmed_docs(f):
                        w.write(doc)
                        docs += 1
            os.makedirs(os.path.dirname(done), exist_ok=True)
            open(done, "w").close()
            return {"file": name, "docs": docs}
        except (OSError, EOFError, ET.ParseError) as e:
            error = f"{type(e).__name__}: {e}"
            time.sleep(10 * (attempt + 1))
    for old in glob.glob(os.path.join(out_dir, f"{prefix}_*.jsonl.zst")):
        os.remove(old)
    return {"file": name, "error": error}


PMC_BUCKET = "pmc-oa-opendata"
PMC_LICENSES = ("CC0", "CC BY", "CC BY-SA")
_REFS = {"references", "reference", "bibliography", "literature cited", "references and notes", "works cited"}


def pmc_keep(meta: dict, licenses=PMC_LICENSES) -> bool:
    return (meta.get("is_pmc_openaccess") is True and meta.get("license_code") in licenses and not meta.get("is_retracted") and not meta.get("is_historical_ocr") and bool(meta.get("text_url")))


def pmc_text(raw: str) -> str:
    lines = raw.splitlines()
    sep = [i for i, ln in enumerate(lines) if re.fullmatch(r"={10,}", ln.strip())]
    title = ""
    if len(sep) >= 2:
        i = sep[1] + 1
        while i < len(lines) and lines[i].strip() and ":" in lines[i]:
            i += 1
        title = next((ln.strip() for ln in lines[i:] if ln.strip()), "")
    start = sep[2] + 1 if len(sep) >= 3 else (sep[1] + 1 if len(sep) == 2 else 0)
    body = lines[start:]
    if len(sep) == 2:
        k = max((j for j, ln in enumerate(body) if re.match(r"(Keywords|License URL|License|Copyright)\b", ln)), default=-1)
        body = body[k + 1 :]
    cut = next((j for j in range(len(body) - 1, int(len(body) * 0.3), -1) if body[j].strip().lower() in _REFS), None)
    if cut is not None:
        body = body[:cut]
    body = [ln for ln in body if not re.fullmatch(r"\s*(Image|Figure|Fig\.|Table)\s*\d+\s*", ln)]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()
    return f"{title}\n\n{text}" if title and not text.startswith(title) else text


def fetch_pmc(out_dir: str, threads: int = 128, licenses=PMC_LICENSES, batch: int = 20_000, log=print) -> dict:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor

    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED, max_pool_connections=threads, retries={"max_attempts": 8}))

    def get(key: str) -> bytes | None:
        try:
            return s3.get_object(Bucket=PMC_BUCKET, Key=key)["Body"].read()
        except Exception:
            return None

    def article(key: str) -> dict | None:
        raw = get(key)
        if raw is None:
            return None
        meta = json.loads(raw)
        if not pmc_keep(meta, licenses):
            return None
        txt = get(meta["text_url"].split(f"{PMC_BUCKET}/", 1)[1].split("?", 1)[0])
        if not txt:
            return None
        text = pmc_text(txt.decode("utf-8", errors="replace"))
        return {"id": meta["pmcid"], "text": text, "pmcid": meta["pmcid"], "pmid": meta.get("pmid"), "license": meta["license_code"], "doi": meta.get("doi")}

    stats = {"listed": 0, "kept": 0, "skipped_batches": 0}
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=PMC_BUCKET, Prefix="metadata/")
    keys: list[str] = []
    done_dir = os.path.join(out_dir, ".done")
    os.makedirs(done_dir, exist_ok=True)
    n_batch = [0]
    with ThreadPoolExecutor(threads) as ex:
        def flush():
            b = n_batch[0]
            n_batch[0] += 1
            marker = os.path.join(done_dir, f"batch_{b:05d}")
            if os.path.exists(marker):
                stats["skipped_batches"] += 1
                keys.clear()
                return
            import glob
            for old in glob.glob(os.path.join(out_dir, f"pmc_{b:05d}_*.jsonl.zst")):
                os.remove(old)
            kept = 0
            with ShardWriter(out_dir, f"pmc_{b:05d}") as w:
                for doc in ex.map(article, keys):
                    if doc and len(doc["text"]) > 500:
                        w.write(doc)
                        kept += 1
            open(marker, "w").close()
            stats["kept"] += kept
            log(json.dumps(stats))
            keys.clear()

        last = None
        for page in pages:
            for o in page.get("Contents", []):
                if not o["Key"].endswith(".json"):
                    continue
                stats["listed"] += 1
                base = o["Key"].rsplit(".", 2)[0]
                if base == last and keys:
                    keys[-1] = o["Key"]
                else:
                    keys.append(o["Key"])
                last = base
            if len(keys) >= batch:
                flush()
        if keys:
            flush()
    return stats
