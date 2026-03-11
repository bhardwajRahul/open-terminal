"""Filesystem abstraction for multi-user mode.

Provides :class:`UserFS`, a unified interface for file operations.

All I/O uses native Python (``aiofiles`` / ``os``).  In multi-user mode
the server process is added to each provisioned user's group, and home
directories are ``chmod 2770`` (setgid + group rwx), so standard file
operations work without subprocess.

After each write operation a ``sudo chown`` call fixes file ownership
so that files belong to the provisioned user, not the server process.
"""

import asyncio
import os
import shutil
import subprocess

import aiofiles
import aiofiles.os


class UserFS:
    """Filesystem operations scoped to an optional OS user.

    *username* is used for ownership fixups after writes (``None`` = stdlib).
    *home* is the user's home directory (default working directory).

    When *username* is set, path validation prevents access to other
    users' home directories (``/home/<other_user>/…``).
    """

    def __init__(self, username: str | None = None, home: str | None = None):
        self.username = username
        self.home = home or os.getcwd()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_path(self, path: str) -> str:
        """Resolve *path* to an absolute path relative to the user's home.

        Absolute paths are normalised in place.  Relative paths are joined
        to ``self.home`` so that they resolve against the user's home
        directory rather than the server process's ``os.getcwd()``.

        In multi-user mode, paths under ``/home/user`` (the server process's
        default home) are automatically rewritten to the provisioned user's
        home directory, since LLMs often hardcode that path.
        """
        if os.path.isabs(path):
            # Swap /home/user → user's actual home when multi-user is active
            if self.username and self.home != "/home/user":
                if path == "/home/user":
                    path = self.home
                elif path.startswith("/home/user/"):
                    path = self.home + path[len("/home/user"):]
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.home, path))

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def _check_path(self, path: str) -> None:
        """Reject paths inside another user's home directory."""
        if not self.username:
            return
        resolved = os.path.abspath(path)
        if not resolved.startswith("/home/"):
            return
        parts = resolved.split("/")  # ['', 'home', '<user>', ...]
        if len(parts) >= 3:
            target_user_dir = parts[2]
            own_home_name = os.path.basename(self.home)
            if target_user_dir != own_home_name:
                raise PermissionError(
                    f"Access denied: {resolved} belongs to another user"
                )

    async def _chown(self, path: str) -> None:
        """Fix ownership of *path* to the provisioned user."""
        if self.username:
            await asyncio.to_thread(
                subprocess.run,
                ["sudo", "chown", f"{self.username}:{self.username}", path],
                check=True, capture_output=True,
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def read(self, path: str) -> bytes:
        """Read raw bytes from *path*."""
        self._check_path(path)
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from *path*."""
        self._check_path(path)
        async with aiofiles.open(path, "r", encoding=encoding, errors="strict") as f:
            return await f.read()

    async def exists(self, path: str) -> bool:
        """Check if *path* exists."""
        self._check_path(path)
        return await aiofiles.os.path.exists(path)

    async def isfile(self, path: str) -> bool:
        """Check if *path* is a regular file."""
        self._check_path(path)
        return await aiofiles.os.path.isfile(path)

    async def isdir(self, path: str) -> bool:
        """Check if *path* is a directory."""
        self._check_path(path)
        return await aiofiles.os.path.isdir(path)

    async def stat(self, path: str) -> dict:
        """Return size, mtime, and type for *path*."""
        self._check_path(path)
        s = await aiofiles.os.stat(path)
        return {
            "size": s.st_size,
            "modified": s.st_mtime,
            "type": "directory" if os.path.isdir(path) else "file",
        }

    async def listdir(self, path: str) -> list[dict]:
        """List directory contents with type, size, and mtime."""
        self._check_path(path)
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
        self._check_path(path)
        return await asyncio.to_thread(lambda: list(os.walk(path)))

    # ------------------------------------------------------------------
    # Write operations (native Python + chown for correct ownership)
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text *content* to *path*, creating parent dirs."""
        self._check_path(path)
        parent = os.path.dirname(path)
        if parent:
            await aiofiles.os.makedirs(parent, exist_ok=True)
            if self.username:
                await self._chown(parent)
        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)
        await self._chown(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw *data* to *path*, creating parent dirs."""
        self._check_path(path)
        parent = os.path.dirname(path)
        if parent:
            await aiofiles.os.makedirs(parent, exist_ok=True)
            if self.username:
                await self._chown(parent)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        await self._chown(path)

    async def mkdir(self, path: str) -> None:
        """Create directory *path* and parents."""
        self._check_path(path)
        await aiofiles.os.makedirs(path, exist_ok=True)
        await self._chown(path)

    async def remove(self, path: str) -> None:
        """Remove *path* (file or directory)."""
        self._check_path(path)
        if os.path.isdir(path):
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await aiofiles.os.remove(path)

    async def move(self, source: str, destination: str) -> None:
        """Move *source* to *destination*."""
        self._check_path(source)
        self._check_path(destination)
        await asyncio.to_thread(shutil.move, source, destination)
        await self._chown(destination)
