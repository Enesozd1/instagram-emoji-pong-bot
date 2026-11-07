# Changelog

## v3.5 + replay fix (repository release)
- Upward-flight prepositioning via `--preposition-up`.
- Automatic replay (`--auto-replay`) using results-button detection with a fallback to the visible `Hits` heading.
- Persistent ADB swipe backend, bounded retries and debug output.
- **Gameplay logic is unchanged** from the final working v3.5 replay-fix script.

## Development milestones
- **Initial prototype:** Desktop screenshot selection, OpenCV tracking and Windows mouse control.
- **ADB transition:** Switched to direct emulator input to avoid hijacking the Windows mouse.
- **v2.x:** Improved game-state detection and paddle tracking.
- **v3.0–3.2:** Experimented with console touch, persistent ADB swipes and gradient-aware detection.
- **v3.3–3.4:** Reduced computer-vision processing cost and improved results-screen detection.
- **v3.5:** Added upward-flight prediction and optional replay.

The project evolved through repeated local experiments. Results depend on the emulator and device performance; development runs ranged widely, with a reported best of **50 hits**.
