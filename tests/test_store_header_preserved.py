# tests/test_store_header_preserved.py
import os
import tempfile

from sqac.store import SqacStore
from sqac.format import read_cartridge


def test_save_preserves_name_description_type():
    st = SqacStore(semantic=False)
    st.add("hello world", key="greeting", validate=False)
    # Seed the original header's extra so save() writes it (extra= path).
    st._header_extra = {"cartridge_type": "memory", "sync_policy": "never"}
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    st.save(p, name="My Cart", description="a desc")
    # simulate a roundtrip via load then save with no args
    st2 = SqacStore.load(p)
    p2 = os.path.join(d, "c2.sqac")
    st2.save(p2)  # no name/description -> must reuse header
    h = read_cartridge(p2).header
    assert h.name == "My Cart"
    assert h.description == "a desc"
    # extra (cartridge_type + arbitrary keys) must survive the roundtrip
    assert h.extra.get("cartridge_type") == "memory"
    assert h.extra.get("sync_policy") == "never"