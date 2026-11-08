"""Shared HTTPS settings.

Older Windows installs can carry an expired root certificate that Python's
OpenSSL picks up from the Windows store, which makes some sites (e.g. ntfy.sh,
Let's Encrypt certificates) fail with "certificate has expired". We trust the
system store PLUS the up-to-date Mozilla bundle from `certifi`, so a valid
chain is found. Certificate verification is never switched off.
"""
from __future__ import annotations

import ssl
from functools import lru_cache


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()          # system trust store, hostname checks, CERT_REQUIRED
    try:
        import certifi  # type: ignore
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass  # certifi missing: the system store alone is used
    # never weaken verification
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx
