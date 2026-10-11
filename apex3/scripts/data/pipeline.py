from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apex3.datapipe.io import ShardWriter, list_files, read_docs, task_args


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def _stem(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def _check_gates(name: str, cfg: dict, a, selected: bool) -> None:
    if cfg.get("license_gate") and not a.license_approved:
        sys.exit(f"{name} needs license approval ({cfg['license_gate']}, spec 11.2). " "Rerun with --license-approved once legal has signed off.")
    if cfg.get("select_required") and not selected:
        sys.exit(f"{name} would take the whole dataset: set config, data_dir, filter or download_patterns in " f"{a.config} from the dataset card first.")


def cmd_download(a):
    from huggingface_hub import snapshot_download

    from apex3.datapipe.sources import load_source_configs

    cfgs = load_source_configs(a.config)
    for name in a.source:
        cfg = cfgs[name]
        if "path" not in cfg:
            print(json.dumps({"source": name, "skipped": "local source (fetched manually)"}))
            continue
        patterns = a.include or cfg.get("download_patterns") or ([f"{cfg['data_dir']}/*"] if cfg.get("data_dir") else None)
        _check_gates(name, cfg, a, selected=bool(patterns))
        out = os.path.join(a.out, name)
        snapshot_download(cfg["path"], repo_type="dataset", local_dir=out, allow_patterns=patterns, max_workers=a.workers)
        print(json.dumps({"source": name, "dir": out, "patterns": patterns}), flush=True)


def cmd_fetch_se(a):
    from apex3.datapipe.manual import SE_SITES, SE_SITES_EXTRA

    sites = a.sites or (SE_SITES_EXTRA if a.extra_sites else SE_SITES)
    task, n = task_args(a.task_id, a.num_tasks)
    todo = [s for s in sites[task::n] if not os.path.exists(os.path.join(a.out, ".done", s))]
    os.makedirs(os.path.join(a.out, ".done"), exist_ok=True)
    work = functools.partial(_fetch_se_one, out=a.out, work=a.work, sort_mem=a.sort_mem)
    with _pool(a.workers) as pool:
        for stats in pool.map(work, todo):
            print(json.dumps(stats), flush=True)


def _fetch_se_one(site, out, work, sort_mem):
    from apex3.datapipe.manual import fetch_stackexchange

    stats = fetch_stackexchange(site, out, os.path.join(work, site), sort_mem=sort_mem)
    open(os.path.join(out, ".done", site), "w").close()
    return stats


def cmd_fetch_pubmed(a):
    from apex3.datapipe.manual import fetch_pubmed_file, pubmed_files

    task, n = task_args(a.task_id, a.num_tasks)
    files = pubmed_files()[task::n]
    total, failed = Counter(), []
    with _pool(a.workers) as pool:
        for stats in pool.map(functools.partial(fetch_pubmed_file, out_dir=a.out), files):
            total.update({"files": 1, "docs": stats.get("docs", 0), "skipped": int(bool(stats.get("skipped")))})
            if "error" in stats:
                failed.append(stats)
    print(json.dumps({"baseline_files": len(files), **total, "failed": failed}))
    if failed:
        sys.exit(f"{len(failed)} PubMed files failed")


def cmd_fetch_pmc(a):
    from apex3.datapipe.manual import fetch_pmc

    print(json.dumps(fetch_pmc(a.out, threads=a.threads, log=lambda m: print(m, flush=True))))


def cmd_ingest(a):
    from apex3.datapipe.sources import iter_file, iter_source, load_source_configs, source_files

    cfg = load_source_configs(a.config)[a.source]
    _check_gates(a.source, cfg, a, bool(cfg.get("config") or cfg.get("data_dir") or cfg.get("filter") or a.local_dir))
    task, n = task_args(a.task_id, a.num_tasks)
    max_docs = a.max_docs // n if a.max_docs else None
    max_bytes = int(a.max_gb * 1e9 / n) if a.max_gb else None
    out = os.path.join(a.out, a.source)
    files = source_files(cfg, task, n, a.local_dir)
    if files is None:
        docs = size = 0
        with ShardWriter(out, f"{a.source}_t{task:05d}") as w:
            for doc in iter_source(cfg, task, n, a.local_dir):
                w.write(doc)
                docs += 1
                size += len(doc["text"].encode())
                if (max_docs and docs >= max_docs) or (max_bytes and size >= max_bytes):
                    break
        print(json.dumps({"source": a.source, "task": task, "docs": docs, "bytes": size}))
        return
    done_dir = os.path.join(out, ".done")
    os.makedirs(done_dir, exist_ok=True)
    stems = {f: f"{hashlib.sha1(f.encode()).hexdigest()[:10]}_{_stem(f)}" for f in files}
    docs = size = 0
    for f in files:
        marker = os.path.join(done_dir, stems[f])
        if os.path.exists(marker):
            with open(marker) as m:
                prev = json.load(m)
            docs, size = docs + prev["docs"], size + prev["bytes"]
    skipped = 0
    for f in files:
        marker = os.path.join(done_dir, stems[f])
        if os.path.exists(marker):
            skipped += 1
            continue
        if (max_docs and docs >= max_docs) or (max_bytes and size >= max_bytes):
            break
        prefix = f"{a.source}_{stems[f]}"
        for old in glob.glob(os.path.join(out, f"{prefix}_*.jsonl.zst")):
            os.remove(old)
        n_docs = n_bytes = 0
        with ShardWriter(out, prefix) as w:
            for doc in iter_file(cfg, f, stems[f]):
                w.write(doc)
                n_docs += 1
                n_bytes += len(doc["text"].encode())
                if (max_docs and docs + n_docs >= max_docs) or (max_bytes and size + n_bytes >= max_bytes):
                    break
        with open(marker, "w") as m:
            json.dump({"file": f, "docs": n_docs, "bytes": n_bytes}, m)
        docs, size = docs + n_docs, size + n_bytes
    print(json.dumps({"source": a.source, "task": task, "files": len(files), "skipped": skipped, "docs": docs, "bytes": size}))


def cmd_decontam(a):
    import numpy as np

    from apex3.datapipe.decontam import build_ngram_table

    texts = [str(row[f]) for path in list_files(a.eval) for row in read_docs(path) for f in a.fields if row.get(f)]
    table = build_ngram_table(texts, a.n)
    np.save(a.out, table)
    print(json.dumps({"texts": len(texts), "ngrams": int(table.size)}))


@functools.lru_cache(maxsize=None)
def _token_table(path):
    import numpy as np

    return np.load(path)


def _decontam_bin_file(path, table_path, eos_id, n, apply):
    import numpy as np
    import zstandard

    from apex3.datapipe.decontam import contaminated_docs, drop_docs
    from apex3.datapipe.tokenize import compress

    marker = path + ".decon"
    if os.path.exists(marker):
        with open(marker) as f:
            return {**json.load(f), "skipped": 1}
    with open(path, "rb") as f:
        arr = np.frombuffer(zstandard.ZstdDecompressor().stream_reader(f).read(), dtype=np.uint32)
    docs, hashes = contaminated_docs(arr, eos_id, _token_table(table_path), n)
    res = {"file": path, "docs": int((arr == eos_id).sum()), "tokens": int(arr.size), "contaminated": int(docs.size)}
    if apply and docs.size:
        kept = drop_docs(arr, eos_id, docs)
        res["removed_tokens"] = int(arr.size - kept.size)
        plain = path[: -len(".zst")]
        kept.tofile(plain + ".tmp")
        os.replace(plain + ".tmp", plain)
        stats_path = plain + ".json"
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            stats.update(docs=stats.get("docs", 0) - int(docs.size), tokens=int(kept.size), bytes=int(stats.get("bytes", 0) * kept.size / max(1, arr.size)))
            with open(stats_path + ".tmp", "w") as f:
                json.dump(stats, f)
            os.replace(stats_path + ".tmp", stats_path)
        compress(plain)
    if apply:
        with open(marker, "w") as f:
            json.dump(res, f)
    uniq, counts = np.unique(hashes, return_counts=True)
    top = np.argsort(-counts)[:200]
    res["hashes"] = dict(zip(uniq[top].tolist(), counts[top].tolist()))
    return res


def cmd_decontam_bin(a):
    import numpy as np

    from apex3.datapipe.decontam import token_ngram_hashes, token_ngram_table
    from apex3.tokens import load_tokenizer, special_ids

    tok = load_tokenizer(a.tokenizer)
    eos = special_ids(tok)["<|endoftext|>"]
    texts = [str(row[f]) for path in list_files(a.eval) for row in read_docs(path) for f in a.fields if row.get(f)]
    token_lists = [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    alpha = np.array([any(c.isascii() and c.isalpha() for c in (tok.id_to_token(i) or "")) for i in range(tok.get_vocab_size())], dtype=bool)
    table = token_ngram_table(token_lists, a.n, a.max_texts, alpha, a.min_alpha)
    os.makedirs(os.path.dirname(os.path.abspath(a.table)), exist_ok=True)
    np.save(a.table, table)
    print(json.dumps({"eval_texts": len(texts), "windows": int(table.size)}), flush=True)
    files = [f for d in a.dir for f in list_files(os.path.join(d, "*.bin.zst"))]
    total, hits = Counter(), Counter()
    work = functools.partial(_decontam_bin_file, table_path=a.table, eos_id=eos, n=a.n, apply=a.apply)
    with _pool(a.workers) as pool:
        for res in pool.map(work, files):
            domain = os.path.basename(os.path.dirname(res["file"]))
            total.update({f"{domain}.{k}": v for k, v in res.items() if isinstance(v, int)})
            hits.update(res.get("hashes", {}))
    print(json.dumps(dict(total), indent=1))
    top = hits.most_common(a.report)
    where = {}
    wanted = np.array([h for h, _ in top], dtype=np.uint64)
    for i, ids in enumerate(token_lists):
        if len(where) == len(top):
            break
        h = token_ngram_hashes(np.asarray(ids, dtype=np.uint32), a.n)
        for k in np.flatnonzero(np.isin(h, wanted)).tolist():
            where.setdefault(int(h[k]), (i, k))
    for h, c in top:
        i, k = where.get(h, (None, None))
        text = tok.decode(token_lists[i][k : k + a.n]) if i is not None else "?"
        print(json.dumps({"windows_matched": c, "eval_text": texts[i][:80] if i is not None else "", "window": text}, ensure_ascii=False))


@functools.lru_cache(maxsize=None)
def _decontaminator(ngram_file, repo_blocklist):
    from apex3.datapipe.decontam import Decontaminator

    return Decontaminator(ngram_file, repo_blocklist)


def _clean_file(path, out_dir, ngram_file, repo_blocklist, pii):
    from apex3.datapipe.clean import clean_doc
    from apex3.datapipe.redact import redact

    decon = _decontaminator(ngram_file, repo_blocklist)
    stats = Counter()
    with ShardWriter(out_dir, _stem(path)) as w:
        for doc in read_docs(path):
            stats["in"] += 1
            doc, reason = clean_doc(doc)
            reason = reason or decon.reason(doc)
            if reason:
                stats[f"drop/{reason}"] += 1
                continue
            doc["text"], counts = redact(doc["text"], secrets=True, pii=pii)
            stats.update({f"redact/{k}": v for k, v in counts.items()})
            w.write(doc)
            stats["kept"] += 1
    open(os.path.join(out_dir, ".done", _stem(path)), "w").close()
    return dict(stats)


def cmd_clean(a):
    task, n = task_args(a.task_id, a.num_tasks)
    os.makedirs(os.path.join(a.out, ".done"), exist_ok=True)
    files = [f for f in list_files(a.inp)[task::n] if not os.path.exists(os.path.join(a.out, ".done", _stem(f)))]
    work = functools.partial(_clean_file, out_dir=a.out, ngram_file=a.decontam, repo_blocklist=a.repo_blocklist, pii=not a.no_pii)
    total = Counter()
    with _pool(a.workers) as pool:
        for stats in pool.map(work, files):
            total.update(stats)
    with open(os.path.join(a.out, f"clean_stats_t{task:05d}.json"), "w") as f:
        json.dump(dict(total), f, indent=1)
    print(json.dumps(dict(total)))


def _sig_worker(args):
    from apex3.datapipe import dedup

    files, shard_ids, work, task, parts = args
    dedup.stage_signatures(files, shard_ids, work, task, dedup.MinHasher(), parts)


def _edges_worker(args):
    from apex3.datapipe import dedup

    dedup.stage_edges(*args)


def _filter_worker(args):
    from apex3.datapipe import dedup

    files, shard, work, out, ngram_file = args
    reject = _decontaminator(ngram_file, None).reason if ngram_file else None
    return dedup.stage_filter(files, shard, work, out, reject)


def cmd_dedup(a):
    from apex3.datapipe.dedup import MinHasher, stage_cluster

    task, n = task_args(a.task_id, a.num_tasks)
    files = list_files(a.inp)
    if a.stage == "sig":
        mine = list(range(len(files)))[task::n]
        jobs = [(files, mine[i :: a.workers], a.work, task * a.workers + i, a.parts) for i in range(a.workers)]
        with _pool(a.workers) as pool:
            list(pool.map(_sig_worker, [j for j in jobs if j[1]]))
    elif a.stage == "edges":
        n_cols = MinHasher().n_bands + 1
        units = [(a.work, b, p) for b in range(n_cols) for p in range(a.parts)][task::n]
        with _pool(a.workers) as pool:
            list(pool.map(_edges_worker, units))
    elif a.stage == "cluster":
        print(json.dumps({"removed": stage_cluster(a.work)}))
    else:
        jobs = [(files, s, a.work, a.out, a.decontam) for s in range(len(files))[task::n]]
        with _pool(a.workers) as pool:
            res = list(pool.map(_filter_worker, jobs))
        print(json.dumps({"kept": sum(r[0] for r in res), "removed": sum(r[1] for r in res), "contaminated": sum(r[2] for r in res)}))


def _partition_worker(args):
    from apex3.datapipe.repo_pack import partition

    partition(*args)


def _pack_worker(args):
    from apex3.datapipe.repo_pack import pack

    return pack(*args)


def cmd_repo_pack(a):
    task, n = task_args(a.task_id, a.num_tasks)
    with _pool(a.workers) as pool:
        if a.stage == "partition":
            files = list_files(a.inp)[task::n]
            jobs = [(files[i :: a.workers], a.work, a.buckets, task * a.workers + i) for i in range(a.workers)]
            list(pool.map(_partition_worker, [j for j in jobs if j[0]]))
        else:
            buckets = sorted(glob.glob(os.path.join(a.work, "bucket_*")))[task::n]
            print(json.dumps({"repo_docs": sum(pool.map(_pack_worker, [(b, a.out, a.max_chars) for b in buckets]))}))


def _tokenize_job(path, out, tokenizer, fim_rate, seed, zst):
    from apex3.datapipe.tokenize import compress, tokenize_file

    if not os.path.exists(out):
        tokenize_file(path, out, tokenizer, fim_rate, seed)
    if zst:
        compress(out)


def cmd_tokenize(a):
    from apex3.tokens import load_tokenizer, special_ids

    task, n = task_args(a.task_id, a.num_tasks)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, os.path.join(a.out, _stem(f) + ".bin")) for f in list_files(a.inp)[task::n]]
    jobs = [(f, out) for f, out in jobs if not os.path.exists(out + ".zst" if a.compress else out)]
    with _pool(a.workers) as pool:
        futures = [pool.submit(_tokenize_job, f, out, a.tokenizer, a.fim_rate, a.seed, a.compress) for f, out in jobs]
        for fut in futures:
            fut.result()
    print(json.dumps({"files": len(jobs), "eos_id": special_ids(load_tokenizer(a.tokenizer))["<|endoftext|>"]}))


def cmd_long(a):
    from apex3.datapipe.tokenize import long_docs_file

    task, n = task_args(a.task_id, a.num_tasks)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, os.path.join(a.out, os.path.basename(f))) for f in list_files(a.inp)[task::n]]
    total = Counter()
    with _pool(a.workers) as pool:
        for s in pool.map(long_docs_file, *zip(*jobs), [a.eos_id] * len(jobs), [a.min_tokens] * len(jobs)):
            total.update(s)
    print(json.dumps({"files": len(jobs), **total, "kept_token_share": total["tokens"] / max(1, total["src_tokens"])}))


def cmd_stats(a):
    total = Counter()
    for path in list_files(os.path.join(a.dir, "*.bin.json")):
        with open(path) as f:
            total.update(json.load(f))
    print(json.dumps({**total, "bytes_per_token": total["bytes"] / max(1, total["tokens"])}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--task-id", type=int, default=None)
        s.add_argument("--num-tasks", type=int, default=None)
        s.add_argument("--workers", type=int, default=os.cpu_count())
        s.add_argument("--local-tasks", type=int, default=None, help="without Slurm: run this command as N parallel tasks on this machine")
        return s

    s = add("download", cmd_download)
    s.add_argument("--config", default="configs/data/sources.json")
    s.add_argument("--source", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--include", nargs="*", default=None, help="file patterns (default: download_patterns in the config)")
    s.add_argument("--license-approved", action="store_true", help="required for sources with a license_gate")
    s = add("fetch-se", cmd_fetch_se)
    s.add_argument("--sites", nargs="*", default=None, help="default: the code and science sites in manual.SE_SITES")
    s.add_argument("--extra-sites", action="store_true", help="the technical sites in manual.SE_SITES_EXTRA instead")
    s.add_argument("--out", required=True)
    s.add_argument("--work", required=True, help="scratch space: Stack Overflow needs ~300 GB while sorting")
    s.add_argument("--sort-mem", default="4G")
    s = add("fetch-pubmed", cmd_fetch_pubmed)
    s.add_argument("--out", required=True)
    s = add("fetch-pmc", cmd_fetch_pmc)
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=128)
    s = add("ingest", cmd_ingest)
    s.add_argument("--config", default="configs/data/sources.json")
    s.add_argument("--source", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--local-dir", default=None, help="files from `download` (default: stream from the Hub)")
    s.add_argument("--max-docs", type=int, default=None)
    s.add_argument("--max-gb", type=float, default=None)
    s.add_argument("--license-approved", action="store_true", help="required for sources with a license_gate")
    s = add("decontam", cmd_decontam)
    s.add_argument("--eval", nargs="+", required=True)
    s.add_argument("--fields", nargs="+", required=True)
    s.add_argument("--n", type=int, default=13)
    s.add_argument("--out", required=True)
    s = add("decontam-bin", cmd_decontam_bin)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--eval", nargs="+", required=True)
    s.add_argument("--fields", nargs="+", required=True)
    s.add_argument("--dir", nargs="+", required=True, help="token shard directories (*.bin.zst)")
    s.add_argument("--table", required=True, help="where to save the token window table (.npy)")
    s.add_argument("--n", type=int, default=13, help="window length in tokens")
    s.add_argument("--max-texts", type=int, default=3, help="drop windows shared by more evaluation texts (templates)")
    s.add_argument("--min-alpha", type=int, default=7, help="drop windows with fewer tokens containing a letter")
    s.add_argument("--apply", action="store_true", help="rewrite shards without the matched documents")
    s.add_argument("--report", type=int, default=30, help="print the most frequently matched windows")
    s = add("clean", cmd_clean)
    s.add_argument("--in", dest="inp", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--decontam", default=None, help="n-gram table from the decontam command")
    s.add_argument("--repo-blocklist", default=None, help="owner/name per line (configs/data/repo_blocklist.txt)")
    s.add_argument("--no-pii", action="store_true", help="redact secrets only")
    s = add("dedup", cmd_dedup)
    s.add_argument("--stage", required=True, choices=["sig", "edges", "cluster", "filter"])
    s.add_argument("--in", dest="inp", nargs="+", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--parts", type=int, default=16)
    s.add_argument("--decontam", default=None, help="filter stage: also drop docs that hit this n-gram table (shards cleaned without --decontam)")
    s = add("repo-pack", cmd_repo_pack)
    s.add_argument("--stage", required=True, choices=["partition", "pack"])
    s.add_argument("--in", dest="inp", nargs="+", default=[])
    s.add_argument("--work", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--buckets", type=int, default=4096)
    s.add_argument("--max-chars", type=int, default=1_000_000)
    s = add("tokenize", cmd_tokenize)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--in", dest="inp", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fim-rate", type=float, default=0.0, help="code only, e.g. 0.5")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--compress", action="store_true", help="store each shard as .bin.zst (storage and transfer; decompress before training)")
    s = add("long", cmd_long)
    s.add_argument("--in", dest="inp", nargs="+", required=True, help="uint32 token shards (*.bin)")
    s.add_argument("--out", required=True)
    s.add_argument("--eos-id", type=int, required=True)
    s.add_argument("--min-tokens", type=int, default=8192, help="8192 for S2 shards, 32768 for S3")
    s = add("stats", cmd_stats)
    s.add_argument("--dir", required=True)
    a = p.parse_args()
    if a.local_tasks and a.task_id is None:
        import subprocess

        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
        procs = [subprocess.Popen(cmd + ["--task-id", str(i), "--num-tasks", str(a.local_tasks)]) for i in range(a.local_tasks)]
        sys.exit(max(128 - c if c < 0 else c for c in [p.wait() for p in procs]))
    a.fn(a)


if __name__ == "__main__":
    main()
