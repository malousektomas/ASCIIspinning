import base64
import io
import math
import os
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from urllib.request import url2pathname

try:
    import numpy as np
    from PIL import Image, ImageGrab, ImageOps, UnidentifiedImageError
except ImportError:
    sys.exit("Missing libraries - install them with:\n\n    pip install pillow numpy\n")

WINDOWS = os.name == "nt"
if WINDOWS:
    import ctypes
    import msvcrt
else:
    import select
    import termios
    import tty

CHAR_ASPECT = 2.0       # terminal characters are roughly twice as high as they are wide
FPS = 30
REVEAL_TIME = 1.3       # conversion animation duration in seconds
RESAMPLE = getattr(Image, "Resampling", Image)

RAMPS = [
    ("classic", " .:-=+*#%@"),
    ("detailed", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"),
    ("blocks", " ░▒▓█"),
]

# (label, axis, sign) - the sign is chosen so the front of the image
# actually moves in the direction shown by the label
DIRECTIONS = [
    ("→ right", "y", -1),
    ("← left", "y", 1),
    ("↑ up", "x", -1),
    ("↓ down", "x", 1),
    ("↻ clockwise", "z", 1),
    ("↺ counterclockwise", "z", -1),
]

SETTINGS = [("size", "Size"), ("speed", "Speed"), ("fps", "FPS"), ("dir", "Direction"), ("thick", "Thickness"),
            ("chars", "Characters"), ("color", "Colors"), ("invert", "Invert"), ("bg", "Background")]

HELP = [("SPACE", "play/pause"), ("←→", "select"), ("↑↓", "change"), ("R", "reset"),
        ("V", "paste from clipboard"), ("N", "new image"), ("S", "save .txt"), ("Q", "quit")]

BANNER = ["▄▀█ █▀ █▀▀ █ █   █▀ █▀█ █ █▄░█",
          "█▀█ ▄█ █▄▄ █ █   ▄█ █▀▀ █ █░▀█"]

MSG_COLORS = {"info": "36", "ok": "1;32", "err": "1;31"}

# side-face normals: left, right, top, bottom (the terminal y-axis points down)
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], float)
LIGHT = np.array([-0.5, -0.6, -1.0]) / np.linalg.norm([-0.5, -0.6, -1.0])   # light from the upper left and front
GLITCH = (85 << 16) | (255 << 8) | 170      # color of "settling" characters during conversion


# ───────────────────────────── helper functions ────────────────────────────

def clamp(value, low, high):
    return max(low, min(high, value))


def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pack_rgb(rgb):
    """RGB (0-255) -> one number; colors are rounded to 16 levels to reduce ANSI codes."""
    q = (np.clip(rgb, 0, 255) * (15 / 255) + 0.5).astype(np.int32) * 17
    return (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]


def gradient(text, start=(0, 200, 255), end=(255, 90, 200)):
    span = max(len(text) - 1, 1)
    out = []
    for i, ch in enumerate(text):
        r, g, b = (round(s + (e - s) * i / span) for s, e in zip(start, end))
        out.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
    return "".join(out) + "\x1b[0m"


def wrap(items, width, sep):
    """Arrange (text, visible length) pairs into lines no wider than width."""
    lines, line, used = [], [], 0
    for text, length in items:
        if line and used + len(sep) + length > width:
            lines.append(sep.join(line))
            line, used = [], 0
        used += length + (len(sep) if line else 0)
        line.append(text)
    if line:
        lines.append(sep.join(line))
    return lines


def place(chars, codes, cols, rows):
    """Center a character grid in a cols x rows area (as in the 3D view)."""
    h, w = chars.shape
    out_chars = np.zeros((rows, cols), np.int16)
    out_codes = np.zeros((rows, cols), np.int32)
    x0, y0 = (cols - w) // 2, (rows - h) // 2
    sx, sy, dx, dy = max(0, -x0), max(0, -y0), max(0, x0), max(0, y0)
    cw, ch = min(w - sx, cols - dx), min(h - sy, rows - dy)
    if cw > 0 and ch > 0:
        out_chars[dy:dy + ch, dx:dx + cw] = chars[sy:sy + ch, sx:sx + cw]
        out_codes[dy:dy + ch, dx:dx + cw] = codes[sy:sy + ch, sx:sx + cw]
    return out_chars, out_codes


def to_ansi(chars, codes, ramp, use_color):
    """Convert character indices (+ colors) to text lines with ANSI colors."""
    lines, escapes = [], {}
    code_rows = codes.tolist() if use_color else None
    for y, row in enumerate(chars.tolist()):
        end = len(row)
        while end and row[end - 1] <= 0:
            end -= 1
        if not use_color:
            lines.append("".join(ramp[k] for k in row[:end]))
            continue
        crow, current, parts = code_rows[y], -1, []
        for x in range(end):
            k = row[x]
            if k > 0 and crow[x] != current:
                current = crow[x]
                esc = escapes.get(current)
                if esc is None:
                    esc = escapes[current] = "\x1b[38;2;%d;%d;%dm" % (
                        current >> 16, (current >> 8) & 255, current & 255)
                parts.append(esc)
            parts.append(ramp[k])
        lines.append("".join(parts))
    return lines


# ───────────────────────────── terminal and keyboard ─────────────────────────

class Terminal:
    """Alternate screen, hidden cursor, and ANSI colors (including Windows console)."""

    def __enter__(self):
        if WINDOWS:
            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel.SetConsoleMode(handle, mode.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        self.write("\x1b]0;ASCII Spin\x07\x1b[?1049h\x1b[?25l\x1b[2J")
        return self

    def __exit__(self, *exc):
        self.write("\x1b[0m\x1b[2J\x1b[?25h\x1b[?1049l")

    @staticmethod
    def write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def size():
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 100, 30
        return max(cols, 30), max(rows, 12)


class Keyboard:
    """Non-blocking key reading. Returns names ("UP", "ENTER", ...) or the characters themselves."""

    NAMES = {"\r": "ENTER", "\n": "ENTER", "\x08": "BACKSPACE", "\x7f": "BACKSPACE",
             "\x1b": "ESC", "\t": "TAB", "\x03": "CTRL_C", "\x16": "CTRL_V"}
    WIN_KEYS = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}
    ANSI_KEYS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}

    def __enter__(self):
        if not WINDOWS:
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if not WINDOWS:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def raw(self):
        """Read everything currently in the keyboard buffer (Windows arrows use two characters)."""
        if WINDOWS:
            chars = []
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in "\x00\xe0":
                    ch += msvcrt.getwch()
                chars.append(ch)
            return chars
        data = b""
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 4096)
            if not chunk:
                break
            data += chunk
        return list(data.decode("utf-8", "replace"))

    def keys(self):
        chars, out, i = self.raw(), [], 0
        while i < len(chars):
            ch = chars[i]
            i += 1
            if len(ch) == 2:
                out.append(self.WIN_KEYS.get(ch[1], ""))
            elif ch == "\x1b" and i < len(chars) and chars[i] in "[O":
                j = i + 1
                while j < len(chars) and not (chars[j].isalpha() or chars[j] == "~"):
                    j += 1
                out.append(self.ANSI_KEYS.get(chars[j], "") if j < len(chars) else "")
                i = j + 1
            else:
                out.append(self.NAMES.get(ch, ch))
        return [k for k in out if k]

    def more(self, quiet=0.06, limit=1.0):
        """Read the rest of pasted text - pasted text arrives in chunks."""
        text, start = [], time.perf_counter()
        last = start
        while time.perf_counter() - last < quiet and time.perf_counter() - start < limit:
            got = self.raw()
            if got:
                text.extend(c for c in got if len(c) == 1 and c.isprintable())
                last = time.perf_counter()
            else:
                time.sleep(0.01)
        return "".join(text)


# ───────────────────────────── image loading ────────────────────────────────

def clipboard_seq():
    """A number Windows increments whenever the clipboard changes (None elsewhere)."""
    if WINDOWS:
        try:
            return ctypes.windll.user32.GetClipboardSequenceNumber()
        except Exception:
            pass
    return None


def clipboard_has_image():
    if not WINDOWS:
        return False
    try:
        user32 = ctypes.windll.user32
        return any(user32.IsClipboardFormatAvailable(f) for f in (2, 8, 15, 17))  # BITMAP, DIB, soubory, DIBV5
    except Exception:
        return False


def grab_clipboard():
    """Return a clipboard image as (PIL.Image, name), or None."""
    try:
        data = ImageGrab.grabclipboard()
    except Exception:
        return None
    if isinstance(data, Image.Image):
        return data, "clipboard image"
    for path in data if isinstance(data, list) else []:      # copied files
        try:
            img = Image.open(path)
            img.load()
            return img, os.path.basename(path)
        except Exception:
            continue
    return None


VIDEO_EXTENSIONS = {".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm", ".wmv"}


class VideoAnimation:
    """Sequential ffmpeg video decoder so the entire video is not loaded into memory."""

    def __init__(self, path):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("ffmpeg and ffprobe are required for video")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate", "-of", "json", path],
            capture_output=True, text=True, check=True, timeout=15)
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise ValueError("video contains no video stream")
        stream = streams[0]
        self.width, self.height = int(stream["width"]), int(stream["height"])
        numerator, denominator = (int(v) for v in stream.get("r_frame_rate", "30/1").split("/", 1))
        self.source_fps = clamp(numerator / max(denominator, 1), 1, 120)
        scale = min(1.0, 1200 / max(self.width, self.height))
        self.out_width = max(2, int(self.width * scale) // 2 * 2)
        self.out_height = max(2, int(self.height * scale) // 2 * 2)
        self.frame_size = self.out_width * self.out_height * 4
        self.path = path
        self.process = None

    def _start(self):
        self.close()
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", self.path, "-f", "rawvideo",
             "-pix_fmt", "rgba", "-vf", f"scale={self.out_width}:{self.out_height}", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def next_frame(self):
        if self.process is None:
            self._start()
        data = self.process.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            self._start()
            data = self.process.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            raise ValueError("could not read a frame from the video")
        return Image.frombytes("RGBA", (self.out_width, self.out_height), data)

    def close(self):
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None


class ImageAnimation:
    """Pillow animated images (GIF, APNG, and animated WebP)."""

    def __init__(self, img):
        self.frames = []
        durations = []
        try:
            count = min(getattr(img, "n_frames", 1), 600)
            for index in range(count):
                img.seek(index)
                durations.append(max(1, int(img.info.get("duration", 100))))
                self.frames.append(img.convert("RGBA").copy())
        except EOFError:
            pass
        if len(self.frames) < 2:
            raise ValueError("animation does not contain enough frames")
        self.source_fps = clamp(1000 / (sum(durations) / len(durations)), 1, 60)
        self.index = 0

    def next_frame(self):
        frame = self.frames[self.index]
        self.index = (self.index + 1) % len(self.frames)
        return frame.copy()

    def close(self):
        pass


def load_from_text(text):
    """Open an image, animated image, or video from a path, URL, or data URI."""
    s = text.strip()
    if s.startswith("& "):                      # PowerShell style: & 'C:\path\image.png'
        s = s[2:].strip()
    if s[:1] in ("'", '"'):
        end = s.find(s[0], 1)
        s = s[1:end] if end > 0 else s[1:]
    s = s.strip()
    if not s:
        raise ValueError("empty input")
    low = s.lower()
    if low.startswith(("http://", "https://")):
        request = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0 ascii-spin"})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(60_000_000)
        img = Image.open(io.BytesIO(data))
        name = os.path.basename(urlparse(s).path) or urlparse(s).netloc
    elif low.startswith("data:image"):
        img = Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1])))
        name = "data URI image"
    else:
        if low.startswith("file:"):
            s = url2pathname(urlparse(s).path)
        path = os.path.abspath(os.path.expanduser(s))
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            animation = VideoAnimation(path)
            try:
                return animation.next_frame(), os.path.basename(path), animation
            except Exception:
                animation.close()
                raise
        img = Image.open(path)
        name = os.path.basename(path)
    img.load()
    animation = ImageAnimation(img) if getattr(img, "n_frames", 1) > 1 else None
    return (animation.next_frame() if animation else img), name, animation


def find_background(img):
    """Find removable background: transparency or a solid-colored area extending from the edges.

    Returns (mask, background_color); mask is an 'L' image (255 = image, 0 = background)."""
    alpha = np.asarray(img.getchannel("A"))
    if (alpha < 128).mean() > 0.02:
        return img.getchannel("A"), None
    small = img.convert("RGB")
    small.thumbnail((256, 256))
    px = np.asarray(small).astype(np.int16)
    edges = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    bg = np.median(edges, axis=0)
    near = np.abs(px - bg).max(axis=2) <= 40
    if np.concatenate([near[0], near[-1], near[:, 0], near[:, -1]]).mean() < 0.8:
        return None, None                       # edges are not uniform -> probably a photo
    filled = np.zeros_like(near)                # fill inward from the edges
    filled[0], filled[-1] = near[0], near[-1]
    filled[:, 0] |= near[:, 0]
    filled[:, -1] |= near[:, -1]
    while True:
        grown = filled.copy()
        grown[1:] |= filled[:-1]
        grown[:-1] |= filled[1:]
        grown[:, 1:] |= filled[:, :-1]
        grown[:, :-1] |= filled[:, 1:]
        grown &= near
        if np.array_equal(grown, filled):
            break
        filled = grown
    if not 0.02 < filled.mean() < 0.97:
        return None, None
    mask = Image.fromarray(np.where(filled, 0, 255).astype(np.uint8)).resize(img.size, RESAMPLE.BILINEAR)
    return mask, tuple(int(v) for v in bg)


class Picture:
    """Loaded image + automatically detected background."""

    def __init__(self, img, name):
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert("RGBA")
        img.thumbnail((1200, 1200))
        self.rgba, self.name = img, name
        self.aspect = img.height / img.width
        self.mask, self.bg = find_background(img)


def stretch(values, selection):
    """Stretch brightness across the full 0-1 range (automatic contrast)."""
    chosen = values[selection]
    low, high = np.percentile(chosen, (1, 99)) if chosen.size else (0.0, 255.0)
    if high - low < 32:
        low, high = 0.0, 255.0
    return np.clip((values - low) / (high - low), 0, 1)


def build_grid(pic, width, keep_bg, invert):
    """Resize the image to a character grid. Returns (height, brightness 0-1, colors, solid cells)."""
    height = max(1, round(width * pic.aspect / CHAR_ASPECT))
    src = pic.rgba
    filt = RESAMPLE.BOX if width < src.width else RESAMPLE.BICUBIC
    use_mask = pic.mask is not None and not keep_bg
    if use_mask:
        im = src.copy()
        im.putalpha(pic.mask)
        arr = np.asarray(im.convert("RGBa").resize((width, height), filt).convert("RGBA"), np.float32)
        rgb, solid = arr[..., :3], arr[..., 3] >= 128
        if not solid.any():
            solid[:] = True
    else:
        flat = Image.new("RGBA", src.size, (0, 0, 0, 255))
        flat.alpha_composite(src)
        rgb = np.asarray(flat.convert("RGB").resize((width, height), filt), np.float32)
        solid = np.ones((height, width), bool)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    if use_mask and pic.bg is not None:
        # removed solid background -> "intensity" is how much the color differs from the background
        ink = np.sqrt(((rgb - np.array(pic.bg, np.float32)) ** 2).sum(axis=2))
        bright = np.clip(ink / max(float(np.percentile(ink[solid], 98)), 1.0), 0, 1)
    else:
        bright = stretch(lum, solid)
        if use_mask and lum[solid].mean() < 80:     # dark logo on a transparent background
            bright = 1 - bright
    if invert:
        bright = 1 - bright
    color = np.clip(rgb + np.clip(80 - lum, 0, None)[..., None], 0, 255)   # slightly brighten dark colors
    return height, bright.astype(np.float32), color.astype(np.float32), solid


# ───────────────────────────── 3D model and rendering ─────────────────────────

class Model:
    """Image ASCII grid and a 3D solid "extruded" from it: front, back, and sides."""

    def __init__(self, pic, width, thickness, keep_bg, invert):
        W = int(width)
        H, bright, color, solid = build_grid(pic, W, keep_bg, invert)
        A, T = CHAR_ASPECT, float(thickness)
        self.W, self.H, self.T = W, H, T
        self.bright, self.color, self.solid = bright, color, solid
        self.dist = T / 2 + 3 * max(W, H * A)                       # camera distance
        self.reveal_at = 0.75 * np.arange(H)[:, None] / max(H - 1, 1) + 0.25 * np.random.random((H, W))

        # front / back: 2x3 points per character to avoid gaps in the rotated image
        jj, ii = np.nonzero(solid)
        ox, oy = np.meshgrid([0.25, 0.75], [1 / 6, 3 / 6, 5 / 6])
        ox, oy = ox.ravel(), oy.ravel()
        self.fx = (ii[:, None] + ox - W / 2).ravel().astype(np.float32)
        self.fy = ((jj[:, None] + oy) * A - H * A / 2).ravel().astype(np.float32)
        self.fb = np.repeat(bright[jj, ii], ox.size)
        self.fc = np.repeat(color[jj, ii], ox.size, axis=0)

        # sides: wherever there is no solid character next to one; slightly inset into the character,
        # otherwise a small rotation would make them protrude into the neighboring cell
        wx, wy, wz, wn, wc = [], [], [], [], []
        if T > 0:
            nz = max(1, math.ceil(T / 0.6))
            zs = (np.arange(nz) + 0.5) * T / nz - T / 2
            pad = np.pad(solid, 1)
            open_sides = [solid & ~pad[1:-1, :-2], solid & ~pad[1:-1, 2:],
                          solid & ~pad[:-2, 1:-1], solid & ~pad[2:, 1:-1]]
            t = np.array([0.25, 0.75])
            for n, mask in enumerate(open_sides):
                j, i = np.nonzero(mask)
                if n < 2:       # left / right side (vertical character edge)
                    x = np.repeat(i + (0.7 if n == 1 else 0.3), 2)
                    y = (j[:, None] + t).ravel()
                else:           # top / bottom side (horizontal character edge)
                    x = (i[:, None] + t).ravel()
                    y = np.repeat(j + (0.7 if n == 3 else 0.3), 2)
                wx.append(np.repeat(x - W / 2, nz))
                wy.append(np.repeat(y * A - H * A / 2, nz))
                wz.append(np.tile(zs, x.size))
                wn.append(np.full(x.size * nz, n, np.int8))
                wc.append(np.repeat(color[j, i], 2 * nz, axis=0))
        self.wx = np.concatenate(wx).astype(np.float32) if wx else np.zeros(0, np.float32)
        self.wy = np.concatenate(wy).astype(np.float32) if wy else np.zeros(0, np.float32)
        self.wz = np.concatenate(wz).astype(np.float32) if wz else np.zeros(0, np.float32)
        self.wn = np.concatenate(wn) if wn else np.zeros(0, np.int8)
        self.wc = np.concatenate(wc) if wc else np.zeros((0, 3), np.float32)

    def flat(self, levels):
        """Plain (unrotated) ASCII art."""
        chars = np.clip(self.bright * (levels - 1) + 0.5, 0, levels - 1).astype(np.int16)
        chars[~self.solid] = 0
        return chars, pack_rgb(self.color)

    def reveal(self, progress, levels):
        """Conversion animation: characters settle from top to bottom out of random characters."""
        chars, codes = self.flat(levels)
        done = progress >= self.reveal_at + 0.15
        busy = (progress >= self.reveal_at) & ~done & self.solid
        noise = np.random.randint(1, levels, size=chars.shape)
        chars = np.where(done, chars, np.where(busy, noise, 0)).astype(np.int16)
        return chars, np.where(busy, GLITCH, codes)

    def render(self, rot, cols, rows, levels):
        """Project the rotated 3D solid onto a cols x rows area (z-buffer, perspective, shading)."""
        T, A, D = self.T, CHAR_ASPECT, self.dist
        r = rot.tolist()
        groups = []
        for side in (-1.0, 1.0):                                  # front (-z) and back (+z) sides
            normal = rot[:, 2] * side
            to_camera = np.array([0.0, 0.0, -D]) - normal * (T / 2)
            if normal @ to_camera <= 1e-9:
                continue                                          # side faces away from the camera
            z = side * T / 2
            X = r[0][0] * self.fx + r[0][1] * self.fy + r[0][2] * z
            Y = r[1][0] * self.fx + r[1][1] * self.fy + r[1][2] * z
            Z = r[2][0] * self.fx + r[2][1] * self.fy + r[2][2] * z
            light = 0.3 + 0.7 * max(0.0, -float(normal[2]))
            groups.append((X, Y, Z, self.fb * light, self.fc * light))
        if self.wx.size:
            X = r[0][0] * self.wx + r[0][1] * self.wy + r[0][2] * self.wz
            Y = r[1][0] * self.wx + r[1][1] * self.wy + r[1][2] * self.wz
            Z = r[2][0] * self.wx + r[2][1] * self.wy + r[2][2] * self.wz
            normals = NORMALS @ rot.T
            n = normals[self.wn]
            facing = n[:, 0] * -X + n[:, 1] * -Y + n[:, 2] * (-D - Z) > 0
            light = (0.3 + 0.7 * np.clip(normals @ LIGHT, 0, 1))[self.wn[facing]]
            groups.append((X[facing], Y[facing], Z[facing], 0.95 * light, self.wc[facing] * light[:, None]))

        chars = np.zeros(rows * cols, np.int16)
        codes = np.zeros(rows * cols, np.int32)
        if groups:
            X, Y, Z, V, C = (np.concatenate(g) for g in zip(*groups))
            depth = Z + D
            scale = (D - T / 2) / depth                           # perspective (front side = 1:1)
            sx = np.floor((cols - self.W) // 2 + self.W / 2 + X * scale).astype(np.int64)
            sy = np.floor((rows - self.H) // 2 + self.H / 2 + Y / A * scale).astype(np.int64)
            ok = (sx >= 0) & (sx < cols) & (sy >= 0) & (sy < rows)
            cell = sy[ok] * cols + sx[ok]
            order = np.lexsort((depth[ok], cell))                 # z-buffer: nearest point in each cell
            sorted_cells = cell[order]
            first = np.ones(order.size, bool)
            first[1:] = sorted_cells[1:] != sorted_cells[:-1]
            pick = np.nonzero(ok)[0][order[first]]
            target = sorted_cells[first]
            chars[target] = np.clip(V[pick] * (levels - 1) + 0.5, 0, levels - 1).astype(np.int16)
            codes[target] = pack_rgb(C[pick])
        return chars.reshape(rows, cols), codes.reshape(rows, cols)


# ───────────────────────────── application ───────────────────────────────

class App:
    def __init__(self):
        self.state = "input"            # "input" = waiting for an image, "view" = ASCII art
        self.running = True
        self.pic = self.model = None
        self.animation = None
        self.size, self.speed, self.thick = 80, 5, 6
        self.fps = 12
        self.frame_acc = 0.0
        self.dir_i = self.ramp_i = self.sel = 0
        self.color, self.invert, self.keep_bg = True, False, False
        self.playing = False
        self.rot = np.eye(3)
        self.front = True               # front facing forward -> render exact 2D ASCII art
        self.reveal_t0 = None
        self.text = ""                  # partially typed path / URL on the intro screen
        self.msg, self.msg_kind, self.msg_until = "", "info", 0.0
        self.clip_seq, self.clip_check = clipboard_seq(), 0.0
        self.term = self.kb = self.screen = None
        self.dirty = True

    # ---------- main loop ----------

    def run(self, initial=""):
        with Terminal() as term, Keyboard() as kb:
            self.term, self.kb = term, kb
            if initial.strip():
                self.load_text(initial)
            last = time.perf_counter()
            while self.running:
                now = time.perf_counter()
                dt, last = min(now - last, 0.1), now
                keys = kb.keys()
                if keys:
                    self.dirty = True
                    if self.state == "input":
                        self.on_input_keys(keys)
                    else:
                        self.on_view_keys(keys)
                self.tick(time.perf_counter(), dt)
                if self.dirty:
                    self.dirty = False
                    self.draw()
                time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - now)))

    def tick(self, now, dt):
        screen = self.term.size()
        if screen != self.screen:
            self.screen = screen
            self.term.write("\x1b[2J")
            self.dirty = True
        if self.msg and now > self.msg_until:
            self.msg = ""
            self.dirty = True
        if self.state == "input":
            if WINDOWS and now - self.clip_check > 0.25:      # newly copied image loads automatically
                self.clip_check = now
                seq = clipboard_seq()
                if seq != self.clip_seq:
                    self.clip_seq = seq
                    self.dirty = True
                    if clipboard_has_image():
                        self.load_clipboard(auto=True)
            return
        if self.reveal_t0 is not None:
            self.dirty = True
        if self.playing:
            if self.animation:
                self.frame_acc += dt * self.fps
                while self.frame_acc >= 1:
                    self.frame_acc -= 1
                    self.pic = Picture(self.animation.next_frame(), self.pic.name)
                    self.rebuild()
            _, axis, sign = DIRECTIONS[self.dir_i]
            self.rot = rotation(axis, sign * self.speed * math.pi / 10 * dt) @ self.rot
            self.front = False
            self.dirty = True

    # ---------- keys ----------

    def on_input_keys(self, keys):
        for k in keys:
            if k == "CTRL_C":
                self.running = False
            elif k == "ESC":
                if self.pic:
                    self.go_view()
                else:
                    self.running = False
            elif k == "ENTER":
                text, self.text = self.text.strip(), ""
                if text:
                    self.load_text(text)
                else:
                    self.load_clipboard()
                return
            elif k == "CTRL_V":
                self.load_clipboard()
                return
            elif k == "BACKSPACE":
                self.text = self.text[:-1]
            elif len(k) == 1 and k.isprintable():
                self.text += k

    def on_view_keys(self, keys):
        typed = [k for k in keys if len(k) == 1 and k.isprintable()]
        if len(typed) >= 4 and len(set(typed)) > 1:            # pasted text = path or URL
            self.load_text("".join(typed) + self.kb.more())
            return
        for k in keys:
            if self.reveal_t0 is not None:                     # a key skips the conversion animation
                self.finish_reveal()
            if k in ("q", "Q", "ESC", "CTRL_C"):
                self.running = False
            elif k == " ":
                self.playing = not self.playing
                self.msg = ""
            elif k == "LEFT":
                self.sel = (self.sel - 1) % len(SETTINGS)
            elif k in ("RIGHT", "TAB"):
                self.sel = (self.sel + 1) % len(SETTINGS)
            elif k in ("UP", "+", "="):
                self.adjust(1)
            elif k in ("DOWN", "-"):
                self.adjust(-1)
            elif k in ("r", "R"):
                self.rot, self.front = np.eye(3), True
            elif k in ("v", "V", "CTRL_V"):
                self.load_clipboard()
                return
            elif k in ("n", "N"):
                self.go_input()
                return
            elif k in ("s", "S"):
                self.save()

    def adjust(self, step):
        key = SETTINGS[self.sel][0]
        if key == "size":
            grown = round(self.size * 1.08) if step > 0 else round(self.size / 1.08)
            self.size = clamp(max(grown, self.size + 2) if step > 0 else min(grown, self.size - 2), 8, 400)
        elif key == "thick":
            self.thick = clamp(self.thick + step, 0, 30)
            if self.front and not self.playing:
                self.flash("Thickness is visible after the image rotates - press SPACE ▶")
        elif key == "invert":
            self.invert = not self.invert
        elif key == "bg":
            self.keep_bg = not self.keep_bg
        else:                                                   # settings that do not require rebuilding the model
            if key == "speed":
                self.speed = clamp(self.speed + step, 1, 20)
            elif key == "fps":
                self.fps = clamp(self.fps + step, 1, 60)
            elif key == "dir":
                self.dir_i = (self.dir_i + step) % len(DIRECTIONS)
            elif key == "chars":
                self.ramp_i = (self.ramp_i + step) % len(RAMPS)
            elif key == "color":
                self.color = not self.color
            if key in ("speed", "dir", "fps") and not self.playing:
                self.flash("Takes effect during playback - press SPACE ▶")
            return
        self.rebuild()

    # ---------- states, messages, loading ----------

    def rebuild(self):
        self.model = Model(self.pic, self.size, self.thick, self.keep_bg, self.invert)

    def go_input(self):
        self.state, self.playing, self.text = "input", False, ""
        if self.animation:
            self.animation.close()
            self.animation = None
        self.clip_seq = clipboard_seq()       # auto-load only images copied from now on
        self.term.write("\x1b[2J")

    def go_view(self):
        self.state = "view"
        self.term.write("\x1b[2J")

    def flash(self, text, kind="info", seconds=4.0):
        self.msg, self.msg_kind = text, kind
        self.msg_until = time.perf_counter() + seconds
        self.dirty = True

    def busy(self, text):
        self.flash(text, "info", 30)
        self.draw()

    def finish_reveal(self):
        self.reveal_t0 = None
        self.flash("✓ Done! Press SPACE to spin the image ▶", "ok", 30)

    def load_clipboard(self, auto=False):
        if not auto:
            self.busy("Reading clipboard...")
        got = grab_clipboard()
        if got:
            self.set_picture(*got)
        elif not auto:
            self.flash("No image in clipboard. Copy one (for example with Win+Shift+S) and try again.", "err", 6)

    def load_text(self, text):
        is_url = text.strip().strip("\"'").lower().startswith(("http://", "https://"))
        self.busy("Downloading image..." if is_url else "Loading image...")
        try:
            img, name, animation = load_from_text(text)
        except FileNotFoundError:
            self.flash("File does not exist: " + text.strip(), "err", 6)
            return
        except UnidentifiedImageError:
            self.flash("This cannot be opened as an image.", "err", 6)
            return
        except Exception as e:
            self.flash(f"Could not load image: {e}", "err", 6)
            return
        self.set_picture(img, name, animation)

    def set_picture(self, img, name, animation=None):
        try:
            picture = Picture(img, name)
        except Exception as e:
            if animation:
                animation.close()
            self.flash(f"Could not process image: {e}", "err", 6)
            return
        if self.animation:
            self.animation.close()
        self.animation = animation
        self.pic = picture
        cols, rows = self.term.size()
        by_height = (rows - 4) * 0.85 * CHAR_ASPECT / self.pic.aspect   # leave room for rotation
        self.size = int(clamp(min((cols - 1) * 0.92, by_height), 8, 400))
        self.fps = round(animation.source_fps) if animation else 12
        self.frame_acc = 0.0
        self.rot, self.front, self.playing = np.eye(3), True, False
        self.rebuild()
        self.msg = ""
        self.reveal_t0 = time.perf_counter()
        self.go_view()

    def save(self):
        ramp = RAMPS[self.ramp_i][1]
        chars, _ = self.model.flat(len(ramp))
        text = "\n".join("".join(ramp[k] for k in row).rstrip() for row in chars.tolist())
        base = "".join(c if c.isalnum() else "_" for c in os.path.splitext(self.pic.name)[0])[:40]
        name = f"ascii_{base or 'image'}_{time.strftime('%H%M%S')}.txt"
        for folder in (os.path.dirname(os.path.abspath(__file__)), os.path.expanduser("~")):
            path = os.path.join(folder, name)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
            except OSError:
                continue
            self.flash(f"Saved: {path}", "ok", 6)
            return
        self.flash("Could not save.", "err")

    # ---------- drawing ----------

    def draw(self):
        if self.state == "input":
            self.draw_input()
        else:
            self.draw_view()

    def draw_input(self):
        cols, rows = self.term.size()
        width = cols - 1
        box = max(24, min(width - 4, 76))
        shown = self.text[-(box - 7):]
        if self.msg:
            hint = f"\x1b[{MSG_COLORS[self.msg_kind]}m{self.msg[:width - 4]}\x1b[0m"
        elif clipboard_has_image():
            hint = "\x1b[32m» Image in clipboard - press ENTER\x1b[0m"
        else:
            hint = ""
        lines = [gradient(row) for row in BANNER] + [
            "\x1b[2mimage or video → ASCII art that spins and plays\x1b[0m",
            "",
            "\x1b[1mHow to insert an image or video:\x1b[0m",
            " • copy it (Win+Shift+S, “Copy image” in a browser, Ctrl+C in File Explorer)",
            "   and press \x1b[1mENTER\x1b[0m" + (" - on Windows it will also load automatically" if WINDOWS else ""),
            " • or drag a file here / paste a path and press \x1b[1mENTER\x1b[0m",
            "   (MP4, MOV, AVI, MKV, and WebM require ffmpeg)",
            "",
            "┌" + "─" * (box - 2) + "┐",
            "│ > " + shown + "\x1b[36m█\x1b[0m" + " " * (box - 6 - len(shown)) + "│",
            "└" + "─" * (box - 2) + "┘",
            hint,
            "",
            "\x1b[2mESC " + ("back" if self.pic else "quit") + "\x1b[0m",
        ]
        top = max(0, (rows - len(lines)) // 2)
        out = ["\x1b[?2026h"]
        for i, line in enumerate(lines):
            out.append(f"\x1b[{top + i + 1};3H{line}\x1b[0m\x1b[K")
        out.append("\x1b[J\x1b[?2026l")
        self.term.write("".join(out))

    def value_text(self, key):
        return {
            "size": str(self.size),
            "speed": str(self.speed),
            "fps": str(self.fps),
            "dir": DIRECTIONS[self.dir_i][0],
            "thick": str(self.thick),
            "chars": RAMPS[self.ramp_i][0],
            "color": "yes" if self.color else "no",
            "invert": "yes" if self.invert else "no",
            "bg": "keep" if self.keep_bg else "auto",
        }[key]

    def status_lines(self, width):
        if self.reveal_t0 is not None:
            pct = min(99, int((time.perf_counter() - self.reveal_t0) / REVEAL_TIME * 100))
            text, color = f"Converting to ASCII... {pct}%", "36"
        elif self.msg:
            text, color = self.msg, MSG_COLORS[self.msg_kind]
        else:
            text, color = f"{self.pic.name} · {self.model.W}x{self.model.H} characters", "2"
        text = text[:max(0, width - 6)]
        sep = (f"\x1b[2m──\x1b[0m \x1b[{color}m{text}\x1b[0m "
               f"\x1b[2m{'─' * max(0, width - len(text) - 4)}\x1b[0m")
        items = [("\x1b[1;32m▶ PLAYING\x1b[0m" if self.playing else "\x1b[1;33m■ PAUSED\x1b[0m", 9)]
        for i, (key, label) in enumerate(SETTINGS):
            value = self.value_text(key)
            if i == self.sel:
                items.append((f"\x1b[1;7m {label} {value} \x1b[0m", len(label) + len(value) + 3))
            else:
                items.append((f"\x1b[2m{label}\x1b[0m {value}", len(label) + len(value) + 1))
        keys = [(f"\x1b[1m{k}\x1b[0m\x1b[2m {d}\x1b[0m", len(k) + len(d) + 1) for k, d in HELP]
        lines = [sep] + wrap(items, width, "  ")
        help_lines = wrap(keys, width, "\x1b[2m · \x1b[0m")
        if self.term.size()[1] - len(lines) - len(help_lines) >= 10:   # omit help in a small window
            lines += help_lines
        return lines

    def draw_view(self):
        cols, rows = self.term.size()
        if self.reveal_t0 is not None and time.perf_counter() - self.reveal_t0 >= REVEAL_TIME:
            self.finish_reveal()
        status = self.status_lines(cols - 1)
        art_cols, art_rows = cols - 1, max(1, rows - len(status))
        ramp = RAMPS[self.ramp_i][1]
        if self.reveal_t0 is not None:
            progress = (time.perf_counter() - self.reveal_t0) / REVEAL_TIME * 1.15
            chars, codes = place(*self.model.reveal(progress, len(ramp)), art_cols, art_rows)
        elif self.front:
            chars, codes = place(*self.model.flat(len(ramp)), art_cols, art_rows)
        else:
            chars, codes = self.model.render(self.rot, art_cols, art_rows, len(ramp))
        out = ["\x1b[?2026h"]
        for y, line in enumerate(to_ansi(chars, codes, ramp, self.color)):
            out.append(f"\x1b[{y + 1};1H{line}\x1b[0m\x1b[K")
        for i, line in enumerate(status):
            out.append(f"\x1b[{art_rows + i + 1};1H{line}\x1b[0m\x1b[K")
        out.append("\x1b[?2026l")
        self.term.write("".join(out))


def main():
    app = App()
    try:
        app.run(" ".join(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
    finally:
        if app.animation:
            app.animation.close()


if __name__ == "__main__":
    main()
