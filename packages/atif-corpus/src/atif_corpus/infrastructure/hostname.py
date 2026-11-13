# SPDX-License-Identifier: Apache-2.0

"""The name a relocatable pass records as ``meta.host``.

A machine is named by its short hostname joined to its DNS domain, or by the
short hostname alone when the domain is missing, ``.local`` or
``.localdomain``. With ``meta.agent`` and ``session_id`` it identifies which
machine holds the session's original transcript and paths.
"""

from __future__ import annotations

import os
import socket
import sys


def machine_hostname() -> str:
    """Return the machine's hostname as recorded in ``meta.host``.

    The domain comes from the canonical name of the hostname (``hostname
    -f``), or from ``USERDNSDOMAIN`` on Windows.
    """
    name = socket.gethostname()
    if sys.platform == "win32":
        domain = os.environ.get("USERDNSDOMAIN", "")
    else:
        try:
            canonical = socket.getaddrinfo(name, None, flags=socket.AI_CANONNAME)[0][3]
        except OSError:
            canonical = ""
        domain = canonical.partition(".")[2]
    short = name.partition(".")[0]
    if not domain or domain.rpartition(".")[2].lower() in ("local", "localdomain"):
        return short
    return f"{short}.{domain}"


__all__ = ["machine_hostname"]
