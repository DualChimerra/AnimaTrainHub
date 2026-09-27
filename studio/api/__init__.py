"""HTTP/API layer — extracted out of studio/server.py starting with PR-5.

Submodules:
    app.py        FastAPI instance + middleware + lifespan wiring
    lifespan.py   startup / shutdown hooks (ensure_dirs / db.init_db / supervisor start-stop / SSE)
    middleware.py _SelectiveGZipMiddleware (skips gzip by path prefix)
    errors.py     4 HTTPException helpers (safe_join_or_400 / validation / data export paths)
    responses.py  shared response constants (EMPTY_STATE)
    static.py     SPAStaticFiles (react-router fallback to index.html)
    main.py       `main()` uvicorn entry point
"""
