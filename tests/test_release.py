"""Tests of tycho.survey.release on a small synthetic catalog: files, manifest hashes, readable parquet/CSV tables,
archives, datasheet, immutability and reproducibility."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
from survey_env import make_root, synthetic_catalog

from tycho.analysis.ricurve import RICurve
from tycho.config import load_config
from tycho.survey.release import LICENSE, MANIFEST, make_release, verify_release

TABLES = ("objects", "ri_runs", "ri_events", "transients")


@pytest.fixture(scope="module")
def pd():
    """pandas, imported when the tests run rather than at collection (pyarrow starts threads, and other test
    modules fork)."""
    import pandas

    return pandas


@pytest.fixture
def env(tmp_path):
    paths = make_root(tmp_path / "root")
    catalog, done = synthetic_catalog(paths)
    yield paths, catalog, done
    catalog.close()


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_files_and_manifest(env):
    paths, catalog, done = env
    target = make_release(catalog, paths, "DR0", cfg=load_config(paths.root))
    assert target == paths.releases / "DR0" and target.is_dir()
    names = sorted(p.name for p in target.iterdir())
    expected = sorted([f"{t}.{ext}" for t in TABLES for ext in ("parquet", "csv")]
                      + ["lightcurves.tar.gz", "ricurves.tar.gz", "sky.yaml", "README.md", MANIFEST])
    assert names == expected  # no calibration.json: the survey is not calibrated; no snapshot left behind
    lines = (target / MANIFEST).read_text(encoding="utf-8").splitlines()
    listed = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        listed[name] = digest
    assert set(listed) == set(expected) - {MANIFEST}
    assert all(sha256(target / name) == digest for name, digest in listed.items())
    assert verify_release(target) == []
    assert not list(paths.releases.glob(".*"))  # the temporary build directory is gone


def test_tables_are_readable_and_hold_only_done_objects(pd, env):
    paths, catalog, done = env
    target = make_release(catalog, paths, "DR0")
    objects = pd.read_parquet(target / "objects.parquet")
    assert sorted(objects["object_id"]) == sorted(done)  # failed and queued objects are not observations
    for col in ("task", "region", "seed", "task_cfg_json", "train_lr", "train_steps", "final_test_acc",
                "t_test_acc90", "n_drops", "diverged", "label", "label_score", "classifier", "code_version",
                "lightcurve_file"):
        assert col in objects.columns
    assert set(objects["classifier"]) == {"rules-v1"} and set(objects["code_version"]) == {"test"}
    assert objects["n_drops"].dtype == "Int64" and objects["diverged"].dtype == "boolean"
    assert all(isinstance(json.loads(s), dict) for s in objects["task_cfg_json"])
    stored = catalog.query("SELECT object_id, final_test_acc FROM summaries")
    by_id = dict(zip(objects["object_id"], objects["final_test_acc"]))
    assert all(by_id[r["object_id"]] == pytest.approx(r["final_test_acc"]) for r in stored)

    events = pd.read_parquet(target / "ri_events.parquet")
    assert len(events) == 1 and json.loads(events["subset_json"][0]) == ["x0", "x1", "yhat"]
    assert events["order"][0] == 3 and bool(events["matches_gt"][0]) is True
    transients = pd.read_parquet(target / "transients.parquet")
    assert sorted(transients["kind"]) == ["loss_spike", "pure_synergy"]
    assert transients["event_id"].dtype == "Int64" and transients["event_id"].isna().sum() == 1
    runs = pd.read_parquet(target / "ri_runs.parquet")
    assert list(runs["kind"]) == ["insitu"] and runs["ricurve_file"][0].startswith("ricurves/")
    for table in TABLES:  # CSV mirrors parquet
        assert len(pd.read_csv(target / f"{table}.csv")) == len(pd.read_parquet(target / f"{table}.parquet"))


def test_archives_hold_light_curves_and_ri_curves(pd, env, tmp_path):
    paths, catalog, done = env
    target = make_release(catalog, paths, "DR0")
    objects = pd.read_parquet(target / "objects.parquet")
    with tarfile.open(target / "lightcurves.tar.gz") as tar:
        members = tar.getmembers()
        assert sorted(m.name for m in members) == sorted(f"lightcurves/{oid}.npz" for oid in done)
        assert all(m.mtime == 0 and m.uid == 0 for m in members)
        lc = np.load(io.BytesIO(tar.extractfile(objects["lightcurve_file"].iloc[0]).read()))
        assert {"step", "train_loss", "test_acc"} <= set(lc.files)
    runs = pd.read_parquet(target / "ri_runs.parquet")
    extracted = tmp_path / "ricurve.npz"
    with tarfile.open(target / "ricurves.tar.gz") as tar:
        extracted.write_bytes(tar.extractfile(runs["ricurve_file"][0]).read())
    curve = RICurve.load(extracted)
    original = RICurve.load(catalog.get_object(runs["object_id"][0])["ri_runs"][0]["path"])
    assert curve.subsets == original.subsets and np.array_equal(curve.ri, original.ri)


def test_datasheet(env):
    paths, catalog, done = env
    readme = (make_release(catalog, paths, "DR0") / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Tycho data release DR0")
    assert LICENSE in readme and "nats" in readme and "MANIFEST.sha256" in readme
    assert f"**{len(done)}** done objects" in readme
    for col in ("object_id", "label_score", "subset_json", "step_lo", "severity", "ricurve_file"):
        assert f"`{col}`" in readme  # schema tables
    assert "Observing code versions: test" in readme and "tiny-v0" in readme


def test_calibration_file_is_included_when_present(env):
    paths, catalog, done = env
    cal = paths.reports / "calibration" / "calibration.json"
    cal.parent.mkdir(parents=True)
    cal.write_text('{"format": "tycho-calibration/1"}', encoding="utf-8")
    target = make_release(catalog, paths, "DR1")
    assert (target / "calibration.json").read_text(encoding="utf-8") == cal.read_text(encoding="utf-8")
    assert "calibration.json" in (target / MANIFEST).read_text(encoding="utf-8")


def test_releases_are_immutable_and_reproducible(env):
    paths, catalog, done = env
    target = make_release(catalog, paths, "DR0")
    before = {name: sha256(target / name) for name in ("lightcurves.tar.gz", "ricurves.tar.gz", "objects.parquet")}
    with pytest.raises(FileExistsError):
        make_release(catalog, paths, "DR0")
    again = make_release(catalog, paths, "DR0", overwrite=True)
    assert sha256(again / "lightcurves.tar.gz") == before["lightcurves.tar.gz"]
    assert sha256(again / "ricurves.tar.gz") == before["ricurves.tar.gz"]
    for bad in ("", "../escape", "a/b", "x" * 65, ".hidden"):
        with pytest.raises(ValueError):
            make_release(catalog, paths, bad)


def test_tampering_is_detected(env):
    paths, catalog, done = env
    target = make_release(catalog, paths, "DR0")
    (target / "objects.csv").write_text("tampered\n", encoding="utf-8")
    (target / "extra.txt").write_text("not listed\n", encoding="utf-8")
    (target / "sky.yaml").unlink()
    problems = verify_release(target)
    assert "checksum mismatch: objects.csv" in problems and "missing: sky.yaml" in problems
    assert "not in the manifest: extra.txt" in problems


def test_release_without_light_curves_and_stale_partials(pd, env):
    paths, catalog, done = env
    stale = paths.releases / ".DR2.partial-99999"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x", encoding="utf-8")
    target = make_release(catalog, paths, "DR2", include_lightcurves=False)
    assert not stale.exists() and not (target / "lightcurves.tar.gz").exists()
    objects = pd.read_parquet(target / "objects.parquet")
    assert objects["lightcurve_file"].isna().all() and verify_release(target) == []


def test_missing_light_curve_file_is_reported_not_fatal(pd, env):
    paths, catalog, done = env
    paths.lightcurve_path(done[1]).unlink()
    target = make_release(catalog, paths, "DR0")
    objects = pd.read_parquet(target / "objects.parquet").set_index("object_id")
    assert pd.isna(objects.loc[done[1], "lightcurve_file"]) and objects["lightcurve_file"].notna().sum() == 4
    assert "1 missing on disk" in (target / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------- sky v1 additions
def test_large_archives_are_split_into_complete_parts(tmp_path):
    from tycho.survey import release as rel

    rng = np.random.default_rng(0)
    members = []
    for i in range(7):
        src = tmp_path / f"m{i}.bin"
        src.write_bytes(rng.bytes(30_000))  # incompressible
        members.append((f"ricurves/m{i}.npz", src))
    names = rel._tar(tmp_path / "ricurves.tar.gz", members, part_bytes=70_000)
    assert names == ["ricurves.part001.tar.gz", "ricurves.part002.tar.gz", "ricurves.part003.tar.gz"]
    got = {}
    for name in names:
        assert (tmp_path / name).stat().st_size < 70_000 + 40_000  # over the limit by at most one member
        with tarfile.open(tmp_path / name) as tar:
            for m in tar.getmembers():
                got[m.name] = tar.extractfile(m).read()
    assert got == {m: src.read_bytes() for m, src in members}
    assert not (tmp_path / "ricurves.tar.gz").exists()
    # a small archive keeps its plain name, and building it twice gives identical bytes
    assert rel._tar(tmp_path / "small.tar.gz", members[:1]) == ["small.tar.gz"]
    first = sha256(tmp_path / "small.tar.gz")
    rel._tar(tmp_path / "small.tar.gz", members[:1])
    assert sha256(tmp_path / "small.tar.gz") == first


def test_split_release_verifies_and_documents_its_parts(env, monkeypatch):
    from tycho.survey import release as rel

    paths, catalog, done = env
    monkeypatch.setattr(rel, "ARCHIVE_PART_BYTES", 1)  # every member in its own part
    target = make_release(catalog, paths, "DRS", cfg=load_config(paths.root))
    parts = sorted(p.name for p in target.glob("lightcurves.part*.tar.gz"))
    assert len(parts) >= 2 and not (target / "lightcurves.tar.gz").exists()
    assert verify_release(target) == []
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert parts[0] in readme and "split into" in readme
    members = set()
    for name in parts:
        with tarfile.open(target / name) as tar:
            members |= set(tar.getnames())
    assert members == {f"lightcurves/{oid}.npz" for oid in done}


def test_release_restricted_to_sky_versions(env, pd):
    import sqlite3

    paths, catalog, done = env
    moved = sorted(done)[:3]
    conn = sqlite3.connect(paths.db)
    conn.executemany("UPDATE objects SET sky_version = 'sky-other' WHERE object_id = ?", [(o,) for o in moved])
    conn.commit()
    conn.close()
    cfg = load_config(paths.root)
    only = make_release(catalog, paths, "ONLY", cfg=cfg, sky_versions=["sky-other"])
    objects = pd.read_parquet(only / "objects.parquet")
    assert sorted(objects["object_id"]) == moved and set(objects["sky_version"]) == {"sky-other"}
    runs = pd.read_parquet(only / "ri_runs.parquet")
    assert set(runs["object_id"]) <= set(moved)
    rest = make_release(catalog, paths, "REST", cfg=cfg)
    assert len(pd.read_parquet(rest / "objects.parquet")) == len(done)  # the catalog itself is untouched
    with pytest.raises(ValueError):
        make_release(catalog, paths, "NONE", cfg=cfg, sky_versions=["no-such-sky"])
    assert not (paths.releases / "NONE").exists()
