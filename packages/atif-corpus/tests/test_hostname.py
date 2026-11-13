# SPDX-License-Identifier: Apache-2.0

"""``machine_hostname``: the machine name a relocatable pass records as ``meta.host``."""

from __future__ import annotations

import socket
import sys

import pytest

from atif_corpus.infrastructure.hostname import machine_hostname


def _machine(monkeypatch: pytest.MonkeyPatch, name: str, canonical: str | None) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(socket, "gethostname", lambda: name)

    def getaddrinfo(host: str, *_args: object, **_kw: object) -> list[tuple[object, ...]]:
        assert host == name
        if canonical is None:
            raise socket.gaierror(socket.EAI_NONAME)
        return [(None, None, None, canonical, None)]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize(
    ("name", "canonical", "expected"),
    [
        ("ws-01", "ws-01.corp.example.com", "ws-01.corp.example.com"),
        ("ws-01.corp.example.com", "ws-01.corp.example.com", "ws-01.corp.example.com"),
        ("macbook", "macbook.local", "macbook"),
        ("box", "box.LocalDomain", "box"),
        ("box", "box", "box"),
        ("box", None, "box"),
    ],
)
def test_the_domain_joins_the_short_name_unless_missing_or_local(
    monkeypatch: pytest.MonkeyPatch, name: str, canonical: str | None, expected: str
) -> None:
    _machine(monkeypatch, name, canonical)
    assert machine_hostname() == expected


@pytest.mark.parametrize(
    ("domain", "expected"), [("CORP.EXAMPLE.COM", "DESKTOP-1.CORP.EXAMPLE.COM"), ("", "DESKTOP-1")]
)
def test_windows_takes_the_domain_from_userdnsdomain(
    monkeypatch: pytest.MonkeyPatch, domain: str, expected: str
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(socket, "gethostname", lambda: "DESKTOP-1")
    monkeypatch.setenv("USERDNSDOMAIN", domain)
    assert machine_hostname() == expected
