"""One-off: delete the three Session 10 lessons retired on 2026-10-19.

195 docker-compose-cvat, 196 apis-and-http, 291 deploy-on-hugging-face, all in
module #35 of course 15. Their Markdown is in git history (commit c6a8c56).
Run by the owner from `courseware/` (the key comes from `.env`):

    uv run --with requests python tools/delete_s10_old_lessons.py
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[2] / "mlarena-sdk"))
import mlarena  # noqa: E402

for line in (HERE / ".env").read_text().splitlines():
    if line.startswith("MLARENA_API_KEY="):
        os.environ.setdefault("MLARENA_API_KEY", line.split("=", 1)[1].strip())

client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"],
                         base_url="https://ml-arena.com")
for lesson_id in (195, 196, 291):
    print(lesson_id, client.delete_lesson(lesson_id))
