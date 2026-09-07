"""The lab's ontology section: discovery, variants, and the routes behind them.

The empty-state test is the load-bearing one. A fresh clone has never run
`python -m blockgen.ontology`, and the page has to say "build one" rather than
show a traceback -- the same contract every other lab route is held to.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from blockgen.ontology import mine, minecraft
from tools.lab import api as lab_api
from tools.lab import ontology as lab_ontology
from blockgen.utils.data import Structure


@pytest.fixture
def catalog_dir(tmp_path, monkeypatch):
    """A one-catalog ontology directory, mined from a trivial synthetic build."""
    ids = np.zeros((4, 4, 4), dtype=np.int32)
    ids[:, 0, :] = 1                       # stone floor
    ids[0, 1:4, 0] = 17                    # oak column
    structure = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    stats = mine.mine_corpus([structure] * 20, corpus="synthetic",
                             min_support=1, min_pair_count=1, top_neighbors=25)
    minecraft.build_catalog(stats, min_support=1).save(tmp_path / "minecraft_synth.json")
    monkeypatch.setattr(lab_ontology, "ONTOLOGY_DIRS", (tmp_path,))
    lab_ontology._palette_symbols.cache_clear()
    return tmp_path


def test_lists_what_is_on_disk(catalog_dir):
    rows = lab_ontology.list_catalogs()
    assert [r["id"] for r in rows] == ["minecraft_synth"]
    assert rows[0]["corpus"] == "synthetic" and rows[0]["n"] > 0


def test_overview_carries_schema_provenance_and_prompt_size(catalog_dir):
    view = lab_ontology.overview()
    assert set(view["sources"]) == {"mined", "asset", "authored", "derived"}
    assert view["prompt"]["tokens"] > 0
    assert len(view["parts"]) == view["catalog"]["n"]
    # Every prompt field must be declared in the schema, or the page's provenance
    # header would show a blank source for a column the model actually reads.
    names = {s["name"] for s in view["schema"]}
    assert set(view["prompt_fields"]) <= names


def test_variants_are_derived_not_stored(catalog_dir):
    mined = lab_ontology.overview(variant="mined")
    shuffled = lab_ontology.overview(variant="shuffled")
    assert mined["prompt"]["chars"] == shuffled["prompt"]["chars"]
    assert [p["id"] for p in mined["parts"]] == [p["id"] for p in shuffled["parts"]]
    assert list(catalog_dir.glob("*.json")) == [catalog_dir / "minecraft_synth.json"]
    with pytest.raises(ValueError):
        lab_ontology.get_catalog(None, "bogus")


def test_part_detail_groups_by_source_and_flags_palette_neighbours(catalog_dir):
    detail = lab_ontology.part_detail("oak_log")
    assert detail["id"] == "oak_log"
    assert "mined" in detail["by_source"]
    assert all(isinstance(n["in_palette"], bool) for n in detail["neighbors"])
    with pytest.raises(KeyError):
        lab_ontology.part_detail("not_a_block")


def test_swatch_returns_real_texture_bytes(catalog_dir):
    from blockgen.ontology.appearance import DEFAULT_PACK_DIR

    if not DEFAULT_PACK_DIR.is_dir():
        pytest.skip("no texture pack fetched")
    body, ctype = lab_ontology.swatch("oak_planks")
    assert ctype == "image/png" and body[:4] == b"\x89PNG"


def test_no_ontology_is_an_empty_state_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_ontology, "ONTOLOGY_DIRS", (tmp_path / "nothing",))
    with pytest.raises(FileNotFoundError):
        lab_ontology.overview()
    assert lab_ontology.list_catalogs() == []


# --- over HTTP -------------------------------------------------------------
@pytest.fixture
def server():
    srv = lab_api.serve("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def test_routes_answer(server, catalog_dir):
    status, body = _get(f"{server}/api/ontology")
    assert status == 200
    payload = json.loads(body)
    assert payload["catalog"]["domain"] == "minecraft"

    status, body = _get(f"{server}/api/ontology/prompt?variant=shuffled")
    assert status == 200 and json.loads(body)["chars"] > 0

    status, body = _get(f"{server}/api/ontology/part/oak_log")
    assert status == 200 and json.loads(body)["id"] == "oak_log"

    status, body = _get(f"{server}/ontology")
    assert status == 200 and b"Ontology" in body


def test_bad_requests_are_4xx_with_a_reason(server, catalog_dir):
    # A name no block family matches is a bad request; a real block whose texture
    # is simply not on disk is the 404. Keeping them apart is what lets the page
    # say "fetch the texture pack" instead of "you typed it wrong".
    for path, code in (("/api/ontology?variant=bogus", 400),
                       ("/api/ontology/part/nope", 404),
                       ("/api/ontology/swatch/not_a_block_at_all", 400)):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{server}{path}")
        assert exc.value.code == code
        assert json.loads(exc.value.read())["error"]
