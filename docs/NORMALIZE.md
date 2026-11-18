# Multilingual Key Normalization

SQAC's lexical/fuzzy tier uses character-trigram BSC encoding, which is
language-agnostic but naive about accented Latin, function words, and
morphological affixes — recall confidence drooped on affix-heavy and
code-switched text. `sqac/textnorm.py` adds an **opt-in, composable
pre-encoding normalization pipeline** in the key tokenizer (used for the
exact tier's canonical form and for everything the fuzzy tier sees).

Enable it per process with one env var — **off by default; byte-identical
behavior to before when unset**:

```bash
export SQAC_NORMALIZE="unicode_fold,stopwords:en,affix:en"
```

## Steps (applied left to right)

| Step | What it does |
|------|--------------|
| `unicode_fold` | NFKD decomposition; combining marks stripped (`naïve` → `naive`, `café` → `cafe`) |
| `stopwords:<lang>` | Drops function words. Built-in: `en` (conservative high-frequency list). Anywhere else: `stopwords:file=<path>` (UTF-8, one word per line, `#` comments) |
| `affix:<lang>` | Light rule-based suffix stripping. `en`: strips `-ing -ed -es -s -ly` when the stem stays ≥ 3 chars. Anywhere else: `affix:file=<path>` (one suffix per line, longest first) |

Examples:

```bash
SQAC_NORMALIZE="unicode_fold"                # accented Latin, minimal risk
SQAC_NORMALIZE="unicode_fold,stopwords:en"   # English noise-grams downweighted
SQAC_NORMALIZE="stopwords:file=es.txt,affix:file=affixes-es.txt"   # e.g. Spanish
```

## How it behaves with existing cartridges

Normalization is **additive and fingerprint-stable**:

- Old entries still load and score — vectors live in the same space; only
  cross-type pairs degrade gracefully.
- The exact tier's canonical form (`key_norm`) uses the same pipeline at
  write and at query time, so exact recall works unchanged.
- **Cartridges are not rewritten** until you compact them; no migration.

For the semantic tier, pointing SQAC at a **user-supplied multilingual
static embedding table** already works today via `SQAC_STATIC_MODEL_DIR`:
set it to a directory containing `model.safetensors` + `tokenizer.json` and
the `static-*` semantic encoder builds from those weights (e.g. a
multilingual potion variant) — zero GPU, zero torch.

## Where it plugs in

`SQAC_NORMALIZE` → `sqac.textnorm.from_env()` → passed by the MCP server
and the HTTP server (rack construction) into every store created or mounted
thereafter: `SqacStore(..., normalizer=...)` / `SqacStore.load(...,
normalizer=...)` → `BSCEncoder(..., normalizer=...)` applies it before the
trigram bundling. `store.normalize()` (exact tier) likewise honors the
instance normalizer. Not set → default encoding, exactly as before.
