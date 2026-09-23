"""
LAUNCHER for the reviewer UI.  Start the UI with THIS file, not with `chainlit run`.

WHY THIS FILE EXISTS
--------------------
`chainlit run app.py` was serving a blank page.  The HTML loaded (HTTP 200) but
every /assets/*.js and *.css came back HTTP 500, so the browser had nothing to
render.

The cause is one line inside Chainlit itself - chainlit/cli/__init__.py:11:

    nest_asyncio.apply()

nest_asyncio monkey-patches asyncio's internals so a loop can be re-entered.
Those internals changed in Python 3.12+, and on Python 3.14 the patch breaks
`asyncio.current_task()`.  sniffio uses current_task() to work out which async
library is running; when it returns None, sniffio says "no async library", and
anyio raises:

    anyio.NoEventLoopError: Not currently running on any asynchronous event loop

Starlette's FileResponse calls `anyio.to_thread.run_sync(os.stat, path)` to stat
the file before sending it.  That is the call that dies - which is exactly why
STATIC FILES failed while the HTML page (built in memory, no threadpool) worked.

Proved it rather than guessed it: a two-route FastAPI app returning a
FileResponse gave HTTP 200; adding `nest_asyncio.apply()` and changing NOTHING
else gave HTTP 500 with the same NoEventLoopError.

THE FIX
-------
Only the chainlit CLI calls nest_asyncio.apply().  The Chainlit app itself does
not need it.  So we mount the Chainlit ASGI app into a plain FastAPI app and run
it with uvicorn directly - the CLI module is never imported, so the monkey-patch
never happens.

START:
    ./serve.sh                       (or)
    .venv/bin/python serve.py
"""
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Chainlit resolves the target relative to the process working directory, so
# anchor it to THIS folder - the same rule we use everywhere else in the project.
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "app.py")

# NOTE: importing chainlit.utils is safe. Importing chainlit.cli is what breaks.
from chainlit.utils import mount_chainlit  # noqa: E402

app = FastAPI(title="Claims Reviewer (host)")


@app.get("/", include_in_schema=False)
def home():
    """Send people who type the bare host straight into the reviewer."""
    return RedirectResponse("/reviewer")


# Chainlit needs a real mount path; it cannot own "/" itself, hence the redirect.
mount_chainlit(app=app, target=TARGET, path="/reviewer")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8300, log_level="warning")
