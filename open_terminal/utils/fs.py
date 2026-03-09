"""Filesystem abstraction for multi-user mode.

Provides :class:`UserFS`, a unified interface for file operations that
transparently routes through ``sudo -u`` when a username is set, or
falls back to standard Python I/O otherwise.  Endpoints receive a
``UserFS`` instance via FastAPI dependency injection and never branch
on the active mode.
"""

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
from typing import Optional

import aiofiles
import aiofiles.os


def _sudo_cmd(username: str) -> list[str]:
    """Base sudo prefix for running a command as *username*."""
    return ["sudo", "-u", username, "--"]


class UserFS:
    """Filesystem operations scoped to an optional OS user.

    When *username* is ``None``, all operations use standard Python
    stdlib / aiofiles (single-user mode).  When *username* is set,
    operations are executed via ``sudo -u <username>`` so that file
    ownership and permissions are correct.

    *home* is the user's home directory, used as the default working
    directory.  Falls back to ``os.getcwd()`` when unset.
    """

    def __init__(self, username: str | None = None, home: str | None = None):
        self.username = username
        self.home = home or os.getcwd()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def read(self, path: str) -> bytes:
        """Read raw bytes from *path*."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["cat", path],
                capture_output=True,
            )
            if result.returncode != 0:
                err = result.stderr.decode().strip()
                if "No such file" in err:
                    raise FileNotFoundError(err)
                raise PermissionError(err)
            return result.stdout
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from *path*."""
        raw = await self.read(path)
        return raw.decode(encoding)

    async def exists(self, path: str) -> bool:
        """Check if *path* exists."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["test", "-e", path],
                capture_output=True,
            )
            return result.returncode == 0
        return await aiofiles.os.path.exists(path)

    async def isfile(self, path: str) -> bool:
        """Check if *path* is a regular file."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["test", "-f", path],
                capture_output=True,
            )
            return result.returncode == 0
        return await aiofiles.os.path.isfile(path)

    async def isdir(self, path: str) -> bool:
        """Check if *path* is a directory."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["test", "-d", path],
                capture_output=True,
            )
            return result.returncode == 0
        return await aiofiles.os.path.isdir(path)

    async def stat(self, path: str) -> dict:
        """Return size and mtime for *path*."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["stat", "-c", "%s %Y %F", path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise FileNotFoundError(result.stderr.strip())
            parts = result.stdout.strip().split(None, 2)
            return {
                "size": int(parts[0]),
                "modified": float(parts[1]),
                "type": "directory" if "directory" in parts[2] else "file",
            }
        s = await aiofiles.os.stat(path)
        return {
            "size": s.st_size,
            "modified": s.st_mtime,
            "type": "directory" if os.path.isdir(path) else "file",
        }

    async def listdir(self, path: str) -> list[dict]:
        """List directory contents with type, size, and mtime."""
        if self.username:
            # Use find for reliable parsing (handles filenames with spaces)
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + [
                    "find", path, "-maxdepth", "1", "-mindepth", "1",
                    "-printf", "%f\\t%y\\t%s\\t%T@\\n",
                ],
                capture_output=True, text=True,
            )
            entries = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t", 3)
                if len(parts) == 4:
                    entries.append({
                        "name": parts[0],
                        "type": "directory" if parts[1] == "d" else "file",
                        "size": int(parts[2]),
                        "modified": float(parts[3]),
                    })
            return sorted(entries, key=lambda e: e["name"])

        def _list_sync():
            entries = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                try:
                    s = os.stat(full)
                    entries.append({
                        "name": name,
                        "type": "directory" if os.path.isdir(full) else "file",
                        "size": s.st_size,
                        "modified": s.st_mtime,
                    })
                except OSError:
                    continue
            return entries
        return await asyncio.to_thread(_list_sync)

    async def walk(self, path: str) -> list[tuple[str, list[str], list[str]]]:
        """Walk directory tree. Returns list of (dirpath, dirnames, filenames)."""
        if self.username:
            result = await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + [
                    "find", path, "-printf", "%y\\t%p\\n",
                ],
                capture_output=True, text=True,
            )
            # Build walk-like structure from find output
            dirs_map: dict[str, tuple[list[str], list[str]]] = {}
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                ftype, fpath = parts
                parent = os.path.dirname(fpath)
                name = os.path.basename(fpath)
                if fpath == path:
                    continue
                if parent not in dirs_map:
                    dirs_map[parent] = ([], [])
                if ftype == "d":
                    dirs_map[parent][0].append(name)
                else:
                    dirs_map[parent][1].append(name)
                # Ensure dirs are in the map even if empty
                if ftype == "d" and fpath not in dirs_map:
                    dirs_map[fpath] = ([], [])
            if path not in dirs_map:
                dirs_map[path] = ([], [])
            return [(d, dns, fns) for d, (dns, fns) in sorted(dirs_map.items())]
        return await asyncio.to_thread(lambda: list(os.walk(path)))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text *content* to *path*, creating parent dirs."""
        if self.username:
            parent = os.path.dirname(path)
            if parent:
                await asyncio.to_thread(
                    subprocess.run,
                    _sudo_cmd(self.username) + ["mkdir", "-p", parent],
                    check=True, capture_output=True,
                )
            await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["tee", path],
                input=content.encode(encoding),
                check=True, capture_output=True,
            )
            return
        parent = os.path.dirname(path)
        if parent:
            await aiofiles.os.makedirs(parent, exist_ok=True)
        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw *data* to *path*, creating parent dirs."""
        if self.username:
            parent = os.path.dirname(path)
            if parent:
                await asyncio.to_thread(
                    subprocess.run,
                    _sudo_cmd(self.username) + ["mkdir", "-p", parent],
                    check=True, capture_output=True,
                )
            await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["tee", path],
                input=data,
                check=True, capture_output=True,
            )
            return
        parent = os.path.dirname(path)
        if parent:
            await aiofiles.os.makedirs(parent, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)

    async def mkdir(self, path: str) -> None:
        """Create directory *path* and parents."""
        if self.username:
            await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["mkdir", "-p", path],
                check=True, capture_output=True,
            )
            return
        await aiofiles.os.makedirs(path, exist_ok=True)

    async def remove(self, path: str) -> None:
        """Remove *path* (file or directory)."""
        if self.username:
            await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["rm", "-rf", path],
                check=True, capture_output=True,
            )
            return
        if os.path.isdir(path):
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await aiofiles.os.remove(path)

    async def move(self, source: str, destination: str) -> None:
        """Move *source* to *destination*."""
        if self.username:
            await asyncio.to_thread(
                subprocess.run,
                _sudo_cmd(self.username) + ["mv", source, destination],
                check=True, capture_output=True,
            )
            return
        await asyncio.to_thread(shutil.move, source, destination)
