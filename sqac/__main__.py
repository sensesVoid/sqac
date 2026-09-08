"""Allow ``python -m sqac`` to work."""
from .cli import main

raise SystemExit(main())
