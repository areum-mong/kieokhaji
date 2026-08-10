# Architecture Overview

This document provides a high‑level overview of the **Kieokhaji‑Portfolio** project architecture.

```
+--- root
|   +--- MOCA                # Flask web application
|       +--- app.py          # Entry point
|       +--- static/         # CSS, JS, images
|       +--- templates/      # Jinja2 HTML templates
|   +--- research/           # Analysis scripts & notebooks
|   +--- data/               # (excluded) raw data files
|   +--- assets/             # Media assets (videos, images)
|   +--- docs/               # Documentation
|   +--- tests/              # Unit tests & CI integration
|   +--- .github/            # GitHub Actions workflows
|   +--- requirements.txt    # Python dependencies
|   +--- README.md           # Project overview
```

*The actual architecture diagram will be added later as an image.*
