from sqac.memid import MonotonicMemId, new_mem_id, decode_mem_id, decode_mem_id_ms


def test_same_ms_is_strictly_increasing():
    g = MonotonicMemId()
    ids = [g.mint() for _ in range(1000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 1000


def test_decode_roundtrips_timestamp():
    mid = new_mem_id()
    ms, rand = decode_mem_id(mid)
    assert isinstance(ms, int) and ms > 1_600_000_000_000  # plausible epoch ms
    assert decode_mem_id_ms(mid) == ms


def test_seed_from_resumes_above_highwater():
    g1 = MonotonicMemId()
    last = g1.mint()
    g2 = MonotonicMemId()
    g2.seed_from(last)
    assert g2.mint() > last


def test_crockford_letters_decode_without_crash():
    # 'Z' (Crockford value 31) would raise under Python int(s, 32)
    mid = new_mem_id()
    ms, _ = decode_mem_id(mid)
    assert ms > 0


def test_randomness_roundtrips():
    # Proves the full 128-bit split (ms << 80 | rand) roundtrips through
    # decode + re-encode; a broken rand mask would fail here.
    mid = new_mem_id()
    ms, rand = decode_mem_id(mid)
    assert MonotonicMemId._encode(ms, rand) == mid


def test_all_z_decodes_to_all_ones():
    # 'Z' -> 31 in Crockford; all-26 'Z' => 32**26 - 1, i.e. max rand and
    # max ms. Pin the Z->31 mapping deterministically.
    expected_ms, expected_rand = (32**26 - 1) >> 80, (1 << 80) - 1
    assert decode_mem_id("Z" * 26) == (expected_ms, expected_rand)
    assert decode_mem_id_ms("Z" * 26) == expected_ms