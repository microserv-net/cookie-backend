"""The command line.

    cookie-backend                 start the server
    cookie-backend pair            issue a pairing code for a new frontend
    cookie-backend devices         list paired frontends
    cookie-backend revoke <name>   remove one
    cookie-backend doctor          check everything, in plain English
    cookie-backend init            write the default configuration and stop
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import __version__
from .auth import DeviceStore
from .config import Config, config_file, load
from .ollama import ModelError, OllamaProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cookie-backend", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", metavar="PATH", help="configuration file to use")
    parser.add_argument("--port", type=int, help="override the configured port")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("pair", help="issue a pairing code")
    sub.add_parser("devices", help="list paired frontends")
    revoke = sub.add_parser("revoke", help="remove a paired frontend")
    revoke.add_argument("name")
    sub.add_parser("doctor", help="check that everything is working")
    sub.add_parser("init", help="write the default configuration and stop")

    args = parser.parse_args(argv)
    path = __import__("pathlib").Path(args.config) if args.config else config_file()
    config = load(path)
    if args.port:
        config.port = args.port

    if args.command == "init":
        print(f"configuration: {path}")
        return 0
    if args.command == "pair":
        return _pair(config)
    if args.command == "devices":
        return _devices(config)
    if args.command == "revoke":
        return _revoke(config, args.name)
    if args.command == "doctor":
        return asyncio.run(_doctor(config))
    return _serve(config)


def _store(config: Config) -> DeviceStore:
    return DeviceStore(config.data_dir / "devices.json")


def _pair(config: Config) -> int:
    code = _store(config).begin_pairing()
    print("Pairing code (valid for ten minutes, single use):\n")
    print(f"    {code}\n")
    print("On the machine running the frontend:\n")
    print(f'    curl -s http://<this-machine>:{config.port}'
          f'{config.base_path}/v1/pair \\')
    print('      -H "content-type: application/json" \\')
    print(f'      -d \'{{"code":"{code}","device_name":"laptop"}}\'\n')
    print("Store the token it returns in COOKIE_BACKEND_TOKEN on that machine.")
    # The code lives in this process, so the server has to be told about it —
    # which is why pairing while the server runs uses the endpoint below.
    print("\nNote: if the server is already running, use its /v1/pair endpoint")
    print("instead; codes are held in memory by the process that issued them.")
    return 0


def _devices(config: Config) -> int:
    devices = _store(config).devices()
    if not devices:
        print("Nothing paired yet. Run `cookie-backend pair`.")
        return 0
    for device in devices:
        seen = "never" if not device.last_seen_at else f"{device.last_seen_at:.0f}"
        print(f"  {device.name:<24} paired {device.created_at:.0f}  last seen {seen}")
    return 0


def _revoke(config: Config, name: str) -> int:
    if _store(config).revoke(name):
        print(f"revoked {name}")
        return 0
    print(f"no device named {name!r}")
    return 1


async def _doctor(config: Config) -> int:
    """The same question the frontend asks: is everything working?"""
    print(f"cookie-backend {__version__}\n")
    provider = OllamaProvider(config)
    problems = 0

    print(f"  config      {config_file()}")
    print(f"  data        {config.data_dir}")
    print(f"  listening   {config.host}:{config.port}{config.base_path}")

    paired = _store(config).devices()
    if paired:
        print(f"  paired      {len(paired)} device(s): "
              f"{', '.join(d.name for d in paired)}")
    else:
        print("  paired      nothing yet — the API is open until you pair")

    if await provider.available():
        print(f"  ollama      up at {config.ollama_endpoint}")
    else:
        print(f"  ollama      NOT REACHABLE at {config.ollama_endpoint}")
        print("              start it with: ollama serve")
        await provider.aclose()
        return 1

    try:
        installed = await provider.installed_models()
    except ModelError as e:
        print(f"  models      {e}")
        await provider.aclose()
        return 1

    for name, role in config.roles.items():
        if role.model in installed:
            print(f"  {name:<11} {role.model} installed, keep_alive {role.keep_alive}")
        else:
            problems += 1
            print(f"  {name:<11} {role.model} MISSING — run: ollama pull {role.model}")

    loaded = await provider.loaded_models()
    resident = sum(m.size_gb for m in loaded)
    print(f"  resident    {resident:.1f} GB of {config.model_memory_gb:.1f} GB budget")
    if resident > config.model_memory_gb:
        problems += 1
        print("              over budget: expect swapping and long pauses")

    await provider.aclose()
    print()
    print("  Everything is ready." if not problems
          else f"  {problems} thing(s) need attention, see above.")
    return 1 if problems else 0


def _serve(config: Config) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Try: pip install -e .", file=sys.stderr)
        return 1
    from .server import create_app

    app = create_app(config)
    print(f"cookie-backend {__version__} on "
          f"http://{config.host}:{config.port}{config.base_path}")
    if _store(config).is_empty():
        print("  nothing paired yet — the API is open until the first device pairs")
        print("  pair one with: POST /v1/pair after `cookie-backend pair`")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
