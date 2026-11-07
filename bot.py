"""Instagram emoji-game bot v3.5 + replay fix (Windows + Android Emulator).

Requirements: py -m pip install numpy opencv-python mss

Workflow: Open the Instagram DM (not game), run script, select the actual
Android display in the frozen screenshot, then click the emoji manually.

Capture: Windows MSS, cropped to selected Android screen.
Control: persistent ADB-shell worker by default, sending the latest requested swipe
without a new adb.exe process each time. Use --controller adb for legacy swipes.

WARNING: This is experimental. No score is guaranteed. Use Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time

import cv2
import mss
import numpy as np

ADB = str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Android' / 'Sdk' / 'platform-tools' / 'adb.exe')
DEFAULT_DEVICE = 'emulator-5554'

# Detection and response tuning, in screen (ROI) coordinates
BALL_DIFF = 42
BALL_SEARCH_TOP = 0.155
BALL_SEARCH_BOTTOM = 0.875
SWIPE_MS = 40
ADB_INTERVAL_S = 0.045
TOUCH_INTERVAL_S = 0.024
TOUCH_DEADBAND_PX = 9   # Android pixels
DELAY_COMPENSATION_S = 0.0
PREDICT_UPWARD = False
TOP_WALL_FRACTION = 0.155
VISION_MAX_WIDTH = 360
_KERNEL = np.ones((3, 3), dtype=np.uint8)


def adb_text(device, *args):
    return subprocess.check_output([ADB, '-s', device, *args], text=True, stderr=subprocess.STDOUT)


def get_display_size(device):
    out = adb_text(device, 'shell', 'wm', 'size')
    sizes = re.findall(r'(\d+)x(\d+)', out)
    if not sizes:
        raise RuntimeError(f'Cannot determine Android screen size: {out}')
    return tuple(map(int, sizes[-1]))


def select_display(screen):
    print('Select ONLY the Android PHONE DISPLAY, not its toolbar or borders.')
    print('Press ENTER after drawing the rectangle.')
    try:
        region = cv2.selectROI('Select entire Android DISPLAY', screen, False, False)
    finally:
        cv2.destroyAllWindows()
    x, y, w, h = map(int, region)
    if w < 100 or h < 200:
        raise RuntimeError('Cancelled or incorrect phone screen selection')
    return x, y, w, h


def background_info(frame):
    h, w = frame.shape[:2]
    points = np.array([
        frame[int(h * yf), int(w * xf)]
        for yf in (0.24, 0.36, 0.50, 0.64, 0.77)
        for xf in (0.04, 0.96)
    ], dtype=np.float32)
    med = np.median(points, axis=0)
    agreement = float(np.mean(np.max(abs(points - med), axis=1) < 31))
    return med, agreement


def small_chat(frame):
    h, w = frame.shape[:2]
    strip = frame[int(.17*h):int(.83*h), int(.05*w):int(.95*w)]
    return cv2.cvtColor(cv2.resize(strip, (72, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def is_chat(frame, chat_signature):
    current = small_chat(frame)
    # Calculate similarity only once every few frames to save processing.
    corr = float(cv2.matchTemplate(current, chat_signature, cv2.TM_CCOEFF_NORMED)[0, 0])
    mad = float(np.mean(abs(current.astype(np.float32) - chat_signature.astype(np.float32))))
    return corr > 0.79 and mad < 19, corr


def paddle_position(frame):
    h, w = frame.shape[:2]
    # Paddle can appear a little higher/lower depending on device.
    y0, y1 = int(h * .79), int(h * .976)
    zone = frame[y0:y1]
    dark = (np.max(zone, axis=2) < 78).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if (.20*w < bw < .70*w and .013*h < bh < .09*h and bw > 2.3*bh):
            candidates.append((x + bw/2, y0 + y + bh/2, bw*bh))
    return max(candidates, key=lambda z: z[2])[:2] if candidates else None


def results_button_center(frame):
    """Return phone-image (x,y) center of the actual white Play again button.

    Uses a wide continuous white band, NOT a missing paddle, as evidence
    that the result screen is displayed. Returns None while playing.
    """
    h, w = frame.shape[:2]
    y0, y1 = int(h * .78), int(h * .95)
    x0, x1 = int(w * .13), int(w * .87)
    zone = frame[y0:y1, x0:x1]
    if zone.size == 0:
        return None
    lo = np.min(zone, axis=2)
    hi = np.max(zone, axis=2)
    white = (lo >= 234) & ((hi - lo) <= 22)
    solid_rows = np.mean(white, axis=1) > .72
    best = (0, -1)
    run_start = None
    for i, present in enumerate(np.r_[solid_rows, False]):
        if present and run_start is None:
            run_start = i
        if not present and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best[1] - best[0] < max(5, int(h * .02)):
        return None
    cy = (best[0] + best[1]) // 2
    # The entire results button is centered horizontally. We still verify
    # that its center lies on the white band before recommending a tap.
    if not white[cy, white.shape[1] // 2]:
        return None
    return (w // 2, y0 + cy)


def results_button_visible(frame):
    return results_button_center(frame) is not None

def results_heading_visible(frame):
    """Identify the large, centered white 'NN Hits' heading on results.

    Unlike Play again, the heading is visible even when the user captures
    only the upper portion of the emulator window. This looks for a broad,
    tall line of white text around the middle; it does not OCR the score.
    """
    h, w = frame.shape[:2]
    x0, x1 = int(w * .10), int(w * .90)
    y0, y1 = int(h * .30), int(h * .79)
    zone = frame[y0:y1, x0:x1]
    if zone.size == 0:
        return False
    lo = np.min(zone, axis=2)
    hi = np.max(zone, axis=2)
    white = (lo >= 215) & ((hi - lo) <= 34)
    # Locate horizontal runs of occupied rows in the main 'Hits' line.
    occupied = np.count_nonzero(white, axis=1) >= max(5, int(w * .018))
    start = None
    for i, present in enumerate(np.r_[occupied, False]):
        if present and start is None:
            start = i
        elif not present and start is not None:
            if i - start >= max(11, round(h * .027)):
                ys, xs = np.nonzero(white[start:i])
                if len(xs) > 0:
                    text_width = int(xs.max() - xs.min() + 1)
                    text_center = (xs.max() + xs.min()) / 2 + x0
                    if (text_width >= w * .31
                            and abs(text_center - w / 2) < w * .18):
                        return True
            start = None
    return False


def replay_target(frame, android_w, android_h, crop_matches_device=True):
    """Return (method, adb_x, adb_y) only for a recognizable results screen.

    First use the white Play again rectangle if it is visible. If it is
    below the Windows capture, detect 'NN Hits' and tap the button's known
    Android-screen position instead (not the cropped Windows coordinates).
    """
    h, w = frame.shape[:2]
    button = results_button_center(frame)
    if button is not None:
        if crop_matches_device:
            return ('button', round(button[0] * android_w / w),
                    round(button[1] * android_h / h))
        return ('button/cropped', android_w // 2, round(android_h * .88))
    if paddle_position(frame) is None and results_heading_visible(frame):
        return ('hits-heading', android_w // 2, round(android_h * .88))
    return None


def estimated_background(frame):
    """Fast gradient model: sample narrow vertical strips and interpolate.

    Downstream vision operates on a small image. Constructing the background
    with OpenCV uint8 resize/blur avoids full-frame NumPy float32 arrays and
    repeated per-row medians, which were expensive in v3.3.
    """
    h, w = frame.shape[:2]
    left0, left1 = max(0, int(w*.035)), max(1, int(w*.085))
    right0, right1 = min(w-1, int(w*.915)), min(w, int(w*.965))
    if left1 <= left0: left1 = left0 + 1
    if right1 <= right0: right1 = right0 + 1
    left = cv2.resize(frame[:, left0:left1], (1, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(frame[:, right0:right1], (1, h), interpolation=cv2.INTER_AREA)
    # Width-2 strips, blurred vertically to ignore brief ball crossings.
    sides = np.concatenate((left, right), axis=1)
    k = max(3, (int(h * .075) | 1))
    sides = cv2.GaussianBlur(sides, (1, k), 0)
    return cv2.resize(sides, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_ball(frame, bg=None):
    """Return candidates in ORIGINAL display coordinates.

    Do costly vision on a resized image, preserving screen coordinates for
    tracking and the controller. Handles smoothly changing background colors.
    """
    h0, w0 = frame.shape[:2]
    scale = min(1.0, VISION_MAX_WIDTH / w0)
    if scale < 1.0:
        w = max(1, round(w0 * scale))
        h = max(1, round(h0 * scale))
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    else:
        img = frame
        h, w = h0, w0
    background = estimated_background(img)
    diff = np.max(cv2.absdiff(img, background), axis=2)
    mask = cv2.inRange(diff, BALL_DIFF, 255)
    mask[:int(h*BALL_SEARCH_TOP)] = 0
    mask[int(h*BALL_SEARCH_BOTTOM):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    sx, sy = w0 / w, h0 / h
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (.035*w < bw < .18*w and .035*w < bh < .18*w):
            continue
        if not (.65 < bw/max(1,bh) < 1.55):
            continue
        if not (.0012*w*w < area < .035*w*w):
            continue
        cx, cy = centroids[i]
        candidates.append((float(cx * sx), float(cy * sy),
                           float(max(bw*sx, bh*sy)*.5), int(area*sx*sy)))
    # Small debug mask is intentional: saving the full mask on every frame
    # would waste CPU, and its purpose is just post-run inspection.
    return candidates, mask


def reflect(value, minimum, maximum):
    width = maximum - minimum
    if width <= 0:
        return value
    t = (value - minimum) % (2*width)
    return minimum + (t if t <= width else 2*width - t)


class Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = self.r = None
        self.vx = self.vy = 0.0
        self.t = None
        self.good = 0
        self.misses = 0
        self.last_measurement = 0.0

    def see(self, candidates, t, width, height):
        if not candidates:
            self.misses += 1
            if self.misses > 7:
                self.reset()
            return False
        if self.x is None:
            # On entry, ball is usually around upper-middle of the field.
            candidate = min(candidates, key=lambda c: abs(c[0]-width*.5) + abs(c[1]-height*.37))
            self.x, self.y, self.r = candidate[:3]
            self.t = t
            self.good = 1
            self.misses = 0
            self.last_measurement = t
            return True

        self.height = height
        dt = t - self.t
        if dt <= 0.003 or dt > 0.40:
            self.misses += 1
            return False
        pred_x = self.x + self.vx * dt
        pred_y = self.y + self.vy * dt
        # Do not pull the tracker to a distant false candidate.
        chosen = min(candidates, key=lambda c: (c[0]-pred_x)**2+(c[1]-pred_y)**2)
        dx, dy = chosen[0] - self.x, chosen[1] - self.y
        max_step = max(width*.075, width*3.0*dt)
        if abs(dx) > max_step or abs(dy) > max_step:
            self.misses += 1
            if self.misses > 7:
                self.reset()
            return False
        inst_vx, inst_vy = dx/dt, dy/dt
        if self.good < 2:
            self.vx, self.vy = inst_vx, inst_vy
        else:
            # Faster adaptation after a real bounce, slower otherwise.
            alpha_x = .95 if self.vx*inst_vx < 0 else .65
            alpha_y = .95 if self.vy*inst_vy < 0 else .65
            self.vx = alpha_x*inst_vx + (1-alpha_x)*self.vx
            self.vy = alpha_y*inst_vy + (1-alpha_y)*self.vy
        self.x, self.y, self.r = chosen[:3]
        self.t = t
        self.good += 1
        self.misses = 0
        self.last_measurement = t
        return True

    def target(self, paddle_y, width):
        if self.x is None:
            return width*.5
        if self.good < 3 or abs(self.vy) < 75:
            return self.x
        r = max(width*.025, min(width*.07, self.r or width*.04))
        if self.vy < -75:
            if not PREDICT_UPWARD or self.good < 5:
                return self.x
            # Predict the eventual paddle intercept after bouncing off the
            # TOP wall, not just the next wall bounce on x. Approximate top
            # boundary is configurable and may vary with Instagram layouts.
            top_y = max(r, self.height * TOP_WALL_FRACTION) if hasattr(self, 'height') else r
            top_y = min(top_y, self.y - 1)
            t_up = max(0, (self.y-top_y)/(-self.vy))
            t_down = max(0, (paddle_y-top_y-r)/(-self.vy))
            horizon = t_up+t_down
            if horizon > 3.6 or horizon < 0.05:
                return self.x
            future_x = reflect(self.x + self.vx*horizon, r, width-r)
            # Long predictions are uncertain: move PART of the way toward
            # the future intercept, leaving room for correction on descent.
            blend = 0.80 if horizon < 1.3 else (0.55 if horizon < 2.3 else 0.35)
            return (1-blend)*self.x + blend*future_x
        impact_t = max(0.0, (paddle_y-self.y-max(0,self.r)) / self.vy)
        if impact_t > 1.6:
            return self.x
        return reflect(self.x + self.vx*(impact_t + DELAY_COMPENSATION_S), r, width-r)


class ConsoleTouch:
    """Persistent finger via emulator console hardware events.

    Console events avoid launching a new ADB process for each swipe.
    Consult https://developer.android.com/studio/run/emulator-console
    """
    def __init__(self, device):
        match = re.fullmatch(r'emulator-(\d+)', device)
        if not match:
            raise RuntimeError('Console mode requires an Android emulator serial like emulator-5554')
        self.sock = socket.create_connection(('127.0.0.1', int(match.group(1))), timeout=2.0)
        self.sock.settimeout(1.0)
        self.buffer = b''
        self.held = False
        self.x = None
        self.y = None
        self.last_sent = 0.0
        self.moves = 0
        greeting = self._until_ok()
        if b'Authentication required' in greeting:
            token_path = Path.home() / '.emulator_console_auth_token'
            if not token_path.is_file():
                raise RuntimeError(f'Emulator console token not found: {token_path}')
            token = token_path.read_text(encoding='utf-8').strip()
            if not token:
                raise RuntimeError('Emulator console authentication token is empty')
            self._command('auth ' + token)
        # This harmless check confirms that console commands are accepted.
        self._command('ping')

    def _until_ok(self):
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            # Each emulator console reply ends with a standalone 'OK' line.
            for terminator in (b'\r\nOK\r\n', b'\nOK\n', b'OK\r\n', b'OK\n'):
                at = self.buffer.find(terminator)
                if at != -1 and (terminator.startswith(b'\r') or terminator.startswith(b'\n') or at == 0 or self.buffer[at-1:at] in (b'\n', b'\r')):
                    end = at + len(terminator)
                    answer, self.buffer = self.buffer[:end], self.buffer[end:]
                    if b'KO:' in answer or b'ERROR' in answer:
                        raise RuntimeError(f'Emulator console rejected a command: {answer.decode(errors="replace")[-250:]}')
                    return answer
            chunk = self.sock.recv(8192)
            if not chunk:
                raise RuntimeError('Emulator console disconnected')
            self.buffer += chunk
        raise TimeoutError('Emulator console did not answer in time')

    def _command(self, message):
        self.sock.sendall((message + '\n').encode('ascii'))
        return self._until_ok()

    def down(self, x, y):
        if self.held:
            return
        self._command('event send '
            f'EV_ABS:ABS_MT_SLOT:0 EV_ABS:ABS_MT_TRACKING_ID:42 '
            f'EV_ABS:ABS_MT_POSITION_X:{x} EV_ABS:ABS_MT_POSITION_Y:{y} '
            'EV_KEY:BTN_TOUCH:1 EV_SYN:0:0')
        self.held = True
        self.x, self.y = x, y
        self.last_sent = time.perf_counter()

    def move(self, x, y):
        now = time.perf_counter()
        if not self.held:
            self.down(x, y)
            return
        if now-self.last_sent < TOUCH_INTERVAL_S or abs(x-self.x) < TOUCH_DEADBAND_PX:
            return
        self._command(f'event send EV_ABS:ABS_MT_POSITION_X:{x} '
                      f'EV_ABS:ABS_MT_POSITION_Y:{y} EV_SYN:0:0')
        self.x, self.y = x, y
        self.last_sent = now
        self.moves += 1

    def release(self):
        if not self.held:
            return
        try:
            self._command('event send EV_ABS:ABS_MT_SLOT:0 '
                          'EV_ABS:ABS_MT_TRACKING_ID:-1 EV_KEY:BTN_TOUCH:0 EV_SYN:0:0')
        finally:
            self.held = False

    def close(self):
        try:
            self.release()
        finally:
            self.sock.close()


class PersistentAdbSwipes:
    """Send ADB swipes through a persistent shell on a separate worker thread.

    A single pending target is replaced by newer targets while a swipe runs,
    preventing the backlog of stale input gestures. This is experimental;
    --controller adb reverts to the verified per-command implementation.
    """
    def __init__(self,device):
        self.proc = subprocess.Popen(
            [ADB,'-s',device,'shell','-T'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            text=True,bufsize=1)
        self.x=self.y=None
        self.moves=0
        self.sent_x=None
        self.last_latency_ms=None
        self._cv=threading.Condition()
        self._request=None
        self._running=True
        self._epoch=0
        self._worker=threading.Thread(target=self._work,daemon=True)
        self._worker.start()

    def down(self,x,y):
        self.x,self.y=x,y

    def move(self,x,y,actual_x=None):
        if self.proc.poll() is not None:
            raise RuntimeError('Persistent ADB shell exited')
        start_x=int(actual_x if actual_x is not None else (self.x if self.x is not None else x))
        x,y=int(x),int(y)
        # Always replace the pending position, even if prior request is running.
        with self._cv:
            self._request=(start_x,x,y,self._epoch)
            self.x,self.y=x,y
            self._cv.notify()

    def _work(self):
        try:
            while True:
                with self._cv:
                    while self._running and self._request is None:
                        self._cv.wait()
                    if not self._running:
                        break
                    start_x,x,y,epoch=self._request
                    self._request=None
                    if epoch != self._epoch:
                        continue
                if abs(start_x-x)<TOUCH_DEADBAND_PX:
                    continue
                t=time.perf_counter()
                cmd=(f'input touchscreen swipe {start_x} {y} {x} {y} '
                     f'{SWIPE_MS}; echo __BOT_SWIPE_DONE__\n')
                assert self.proc.stdin is not None
                self.proc.stdin.write(cmd)
                self.proc.stdin.flush()
                assert self.proc.stdout is not None
                # Wait until this swipe has completed. Do not flood ADB's input queue.
                while True:
                    line=self.proc.stdout.readline()
                    if not line:
                        raise RuntimeError('ADB shell closed while waiting for swipe')
                    if '__BOT_SWIPE_DONE__' in line:
                        break
                self.moves+=1
                self.sent_x=x
                self.last_latency_ms=round((time.perf_counter()-t)*1000)
        except (OSError, RuntimeError, ValueError, AssertionError) as e:
            self._error=str(e)

    def release(self):
        with self._cv:
            self._epoch+=1
            self._request=None
            self.x=self.y=None

    def close(self):
        self.release()
        with self._cv:
            self._running=False
            self._cv.notify()
        self._worker.join(timeout=0.3)
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class AdbSwipes:
    """ADB touchscreen swipes, starting at the paddle's observed location.

    Launch at most one command at a time, and do not queue stale gestures.
    This backend was verified to move Instagram's paddle on this emulator.
    """
    def __init__(self, device):
        self.device = device
        self.x = self.y = None
        self.proc = None
        self.last_sent = 0.0
        self.moves = 0

    def down(self, x, y):
        self.x, self.y = x, y

    def move(self, x, y, actual_x=None):
        now = time.perf_counter()
        if self.x is None:
            self.down(actual_x if actual_x is not None else x, y)
        # Prefer the visually measured paddle position over the previous target.
        start_x = int(actual_x if actual_x is not None else self.x)
        if now - self.last_sent < ADB_INTERVAL_S or abs(start_x - x) < TOUCH_DEADBAND_PX:
            return
        if self.proc is not None and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            [ADB, '-s', self.device, 'shell', 'input', 'touchscreen', 'swipe',
             str(start_x), str(y), str(x), str(y), str(SWIPE_MS)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.x, self.y = x, y
        self.last_sent = now
        self.moves += 1

    def release(self):
        self.x = self.y = None

    def close(self):
        self.release()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()


def main():
    parser = argparse.ArgumentParser(description='Instagram emoji game bot v3.5')
    parser.add_argument('--device', default=DEFAULT_DEVICE)
    parser.add_argument('--controller', choices=('adb-fast','adb','console'), default='adb-fast')
    parser.add_argument('--swipe-ms', type=int, default=40, help='ADB swipe duration in milliseconds (default 40)')
    parser.add_argument('--lead-ms', type=float, default=0, help='Extra horizontal trajectory lead in milliseconds (default 0; try 0 before 110)')
    parser.add_argument('--vision-width', type=int, default=360, help='Maximum width for ball detection (default 360); lower is faster, higher may be more accurate')
    parser.add_argument('--preposition-up', action='store_true', help='Experimentally predict the intercept while the ball is rising')
    parser.add_argument('--top-wall', type=float, default=0.155, help='Estimated upper bounce y as a fraction of the phone display (default 0.155)')
    parser.add_argument('--auto-replay', type=int, default=0, metavar='GAMES', help='Automatically tap Play again until this many total games have finished (0=off)')
    parser.add_argument('--output-dir', default=None, help='Folder for results screenshots (default: a new timestamped folder)')
    args = parser.parse_args()
    global SWIPE_MS, DELAY_COMPENSATION_S, VISION_MAX_WIDTH, PREDICT_UPWARD, TOP_WALL_FRACTION
    if not 10 <= args.swipe_ms <= 250:
        parser.error('--swipe-ms must be between 10 and 250')
    if not 0 <= args.lead_ms <= 200:
        parser.error('--lead-ms must be between 0 and 200')
    if not 220 <= args.vision_width <= 640:
        parser.error('--vision-width must be between 220 and 640')
    if not 0 <= args.auto_replay <= 500:
        parser.error('--auto-replay must be 0 to 500 games')
    if not .08 <= args.top_wall <= .35:
        parser.error('--top-wall must be between 0.08 and 0.35')
    PREDICT_UPWARD = args.preposition_up
    TOP_WALL_FRACTION = args.top_wall
    VISION_MAX_WIDTH = args.vision_width
    SWIPE_MS = args.swipe_ms
    DELAY_COMPENSATION_S = args.lead_ms / 1000.0
    output_dir = Path(args.output_dir) if args.output_dir else Path('emoji_runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Saving result images to:', output_dir.resolve())

    if adb_text(args.device, 'get-state').strip() != 'device':
        raise RuntimeError('Android emulator is not connected via ADB')
    android_w, android_h = get_display_size(args.device)
    print(f'Android screen: {android_w} x {android_h}')
    controller = None
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        print('Open the Instagram chat, but do NOT start the emoji game.')
        input('Press ENTER when ready to select the phone area...')
        full = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
        gx, gy, gw, gh = select_display(full)
        aspect_roi = gw / gh
        aspect_android = android_w / android_h
        crop_matches_device = abs(aspect_roi / aspect_android - 1) <= .10
        if not crop_matches_device:
            print(f'WARNING: Selected region aspect ratio {aspect_roi:.3f} does not match '
                  f'Android display ratio {aspect_android:.3f}. Make sure the entire phone '
                  'screen is visible, with no cropping or toolbar included.')
        region = {'left': monitor['left']+gx, 'top': monitor['top']+gy,
                  'width': gw, 'height': gh}
        def grab_phone():
            return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)

        baseline = grab_phone()
        chat_sig = small_chat(baseline)
        chat_bg, _ = background_info(baseline)
        if args.controller == 'console':
            try:
                controller = ConsoleTouch(args.device)
            except (OSError, RuntimeError, TimeoutError) as ex:
                print('Console control unavailable:', ex)
                print('Falling back to known ADB swipe control.')
                controller = AdbSwipes(args.device)
        elif args.controller == 'adb-fast':
            controller = PersistentAdbSwipes(args.device)
        else:
            controller = AdbSwipes(args.device)
        print('Control backend:', 'persistent ADB shell' if isinstance(controller, PersistentAdbSwipes) else ('experimental console' if isinstance(controller, ConsoleTouch) else 'legacy ADB swipes'))
        print('\nBOT ARMED. Click the emoji in Instagram to start the game.')
        print('Bot does not touch the chat. Press Ctrl+C in PowerShell to stop.')

        tracker = Tracker()
        active = False
        startup_frames = 0
        chat_since = None
        missing_paddle_since = None
        missing_paddle_warned = False
        results_frames = 0
        last_paddle = None
        current_bg = chat_bg
        last_chat_check = 0.0
        chat_like = False
        last_report = time.perf_counter()
        frames = balls = 0
        capture_total = vision_total = 0.0
        last_frame = last_mask = None
        last_prediction = None
        game_number = 0
        finished_games = 0
        replay_sent_at = None
        replay_process = None
        replay_attempts = 0
        replay_waiting = False
        start_times = {}

        try:
            while True:
                capture_start = time.perf_counter()
                frame = grab_phone()
                capture_ms = 1000.0 * (time.perf_counter() - capture_start)
                if isinstance(controller, PersistentAdbSwipes) and hasattr(controller, '_error'):
                    raise RuntimeError('ADB worker error: ' + controller._error)
                now = time.perf_counter()
                background, uniformity = background_info(frame)
                paddle = paddle_position(frame)
                # Avoid expensive similarity calculation every frame.
                if now - last_chat_check > .18:
                    chat_like, _ = is_chat(frame, chat_sig)
                    last_chat_check = now

                if not active:
                    # REPLAY ONLY: a previous tap may not register. Retry at
                    # most three times, but only while the RESULTS SCREEN is
                    # still positively recognized. Never touch active games.
                    if (args.auto_replay and replay_waiting
                            and replay_sent_at is not None
                            and now - replay_sent_at > 1.8
                            and replay_attempts < 4):
                        retry = replay_target(frame, android_w, android_h,
                                              crop_matches_device)
                        if retry is not None:
                            method, tap_x, tap_y = retry
                            replay_process = subprocess.Popen(
                                [ADB, '-s', args.device, 'shell', 'input', 'tap',
                                 str(tap_x), str(tap_y)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            replay_attempts += 1
                            replay_sent_at = now
                            print(f'Replay retry {replay_attempts}/4: '
                                  f'{method} at ({tap_x},{tap_y})')
                    game_bg_diff = float(np.max(abs(background-chat_bg)))
                    candidate_start = (uniformity > .65 and game_bg_diff > 30
                                       and paddle is not None and not chat_like)
                    startup_frames = startup_frames+1 if candidate_start else 0
                    if startup_frames < 2:
                        continue
                    game_number += 1
                    active = True
                    tracker.reset()
                    current_bg = background
                    last_paddle = paddle
                    chat_since = None
                    missing_paddle_since = None
                    missing_paddle_warned = False
                    results_frames = 0
                    last_report = now
                    start_times[game_number] = now
                    replay_sent_at = None
                    replay_attempts = 0
                    replay_waiting = False
                    print(f'GAME #{game_number} DETECTED - playing')

                frames += 1
                capture_total += capture_ms
                last_frame = frame
                if chat_like:
                    chat_since = now if chat_since is None else chat_since
                    controller.release()
                    if now - chat_since > 0.6:
                        print('Chat detected: waiting for next game')
                        active = False
                        startup_frames = 0
                    continue
                chat_since = None

                # Only a positively identified results screen ends the game.
                # Temporary loss of ball or paddle must not reset tracking.
                replay_info = replay_target(frame, android_w, android_h,
                                            crop_matches_device)
                results_frames = results_frames + 1 if replay_info is not None else 0
                if results_frames >= 3:
                    controller.release()
                    finished_games += 1
                    duration = now-start_times.get(game_number, now)
                    end_file = output_dir / f'game_{game_number:03d}_end.png'
                    cv2.imwrite(str(end_file), frame)
                    print(f'Game {game_number} finished after {duration:.1f}s. '
                          f'Results screenshot: {end_file}')
                    active = False
                    startup_frames = 0
                    tracker.reset()
                    last_paddle = None
                    missing_paddle_since = None
                    missing_paddle_warned = False
                    if args.auto_replay and finished_games >= args.auto_replay:
                        print(f'Completed {finished_games} games. Stopping as requested.')
                        break
                    if args.auto_replay:
                        method, x_android, y_android = replay_info
                        replay_process = subprocess.Popen(
                            [ADB, '-s', args.device, 'shell', 'input', 'tap',
                             str(x_android), str(y_android)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        replay_sent_at = now
                        replay_attempts = 1
                        replay_waiting = True
                        print(f'Auto replay: detected {method}; tapping '
                              f'({x_android},{y_android}).')
                    else:
                        print('Waiting for manual replay.')
                    continue

                if paddle is None:
                    missing_paddle_since = now if missing_paddle_since is None else missing_paddle_since
                    if now - missing_paddle_since > 0.8 and not missing_paddle_warned:
                        print('Paddle temporarily not visible; continuing to track game.')
                        missing_paddle_warned = True
                else:
                    last_paddle = paddle
                    missing_paddle_since = None
                    missing_paddle_warned = False

                vision_start = time.perf_counter()
                candidates, mask = detect_ball(frame, background)
                vision_total += 1000.0 * (time.perf_counter() - vision_start)
                last_mask = mask
                seen = tracker.see(candidates, now, gw, gh)
                if seen:
                    balls += 1

                if (last_paddle is not None and tracker.x is not None
                        and tracker.good >= 3 and now - tracker.last_measurement < .16):
                    paddle_x, paddle_y = last_paddle
                    desired = tracker.target(paddle_y, gw)
                    desired = np.clip(desired, gw * .085, gw * .915)
                    target_android_x = int(desired * android_w / gw)
                    target_android_y = int(paddle_y * android_h / gh)
                    # When visible, start from the observed paddle position.
                    # When temporarily hidden, use the last commanded position.
                    if paddle is not None:
                        actual_android_x = int(paddle_x * android_w / gw)
                    else:
                        actual_android_x = controller.x if controller.x is not None else int(paddle_x * android_w / gw)
                    if controller.x is None:
                        controller.down(actual_android_x, target_android_y)
                    controller.move(target_android_x, target_android_y, actual_x=actual_android_x)
                    last_prediction = (desired, paddle_y)

                if now - last_report >= 1.0:
                    actual = 'missing' if paddle is None else str(int(paddle[0]*android_w/gw))
                    print(f'FPS={frames} ball={balls} vx={tracker.vx:.0f} vy={tracker.vy:.0f} '
                          f'wanted={controller.x} sent={getattr(controller, "sent_x", None)} actual={actual} '
                          f'moves={controller.moves} backend={"adb-fast" if isinstance(controller, PersistentAdbSwipes) else ("console" if isinstance(controller, ConsoleTouch) else "adb")}'
                          + (f' swipe_ms={controller.last_latency_ms}' if isinstance(controller, PersistentAdbSwipes) else '')
                          + f' capture_ms={capture_total/max(1,frames):.1f}'
                          + f' vision_ms={vision_total/max(1,frames):.1f}')
                    capture_total = vision_total = 0.0
                    frames = balls = 0
                    controller.moves = 0
                    last_report = now

        except KeyboardInterrupt:
            print('\nStopping bot...')
        finally:
            if last_frame is not None:
                debug = last_frame.copy()
                if tracker.x is not None:
                    cv2.circle(debug, (round(tracker.x), round(tracker.y)),
                               max(5, int(tracker.r or 10)), (0,0,255), 2)
                if last_prediction is not None:
                    px, py = last_prediction
                    cv2.line(debug, (round(px), round(py)-32),
                             (round(px), round(py)+18), (0,210,0), 2)
                cv2.imwrite(str(output_dir / 'debug_tracking.png'), debug)
                if last_mask is not None:
                    cv2.imwrite(str(output_dir / 'debug_mask.png'), last_mask)
            if controller is not None:
                controller.close()
            print('Stopped. Result images and diagnostics are in:', output_dir.resolve())


if __name__ == '__main__':
    main()
