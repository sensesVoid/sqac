import time
from sqac.memid import new_mem_id

def test_format_and_sortability():
    a = new_mem_id()
    time.sleep(0.002)
    b = new_mem_id()
    assert len(a) == 26 and len(b) == 26
    assert a < b  # time-sortable
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in a)
