"""Built-in browser skills seeded at workspace level on first init.

These skills teach the agent HOW to use browser tools effectively.
They are seeded as workspace-level skills (lowest priority, overridable).
"""

from __future__ import annotations

from pathlib import Path

from arion_agent.util.persistence import ensure_directory, seed_file

WEB_NAVIGATION_SKILL = """\
---
name: web-navigation
description: Navigate complex multi-page flows. Use when browsing multi-step websites, handling redirects, or following link chains.
---

# Web Navigation Skill

## Core pattern
1. browser_tab_list() to see which tabs are open and their URLs. Use the tab name that has the page you need, or default if only one tab.
2. browser_action(navigate, tab=default, target=URL) to go to the page (if not already there)
3. browser_snapshot(tab=...) to see the simplified HTML of the page (use the tab from step 1)
4. Read attributes from the snapshot to build CSS selectors
5. browser_action(click, tab=..., target=selector) to interact
6. browser_snapshot(tab=...) again to verify the result

## Building selectors from snapshot output
The snapshot shows simplified HTML with attributes. Use these to target elements:
- Attribute selectors: [data-item-key="value"], [role="button"]
- Tag + attribute: a[href="/path"], input[name="query"]
- Class selectors: .class-name (from class attributes in snapshot)
- ID selectors: #element-id
- Combined: button[type="submit"], nav a[href="/search"]

## Multi-step flows
- Login: snapshot -> find form -> fill fields -> click submit -> snapshot to verify
- Pagination: snapshot -> find "next" link -> click -> snapshot -> repeat
- Modal/dropdown: click trigger -> snapshot to see revealed content -> interact
- Menus: click parent menu -> snapshot again -> sub-items now visible -> click target

## Tips
- When snapshot returns about:blank but you expect a real page, call browser_tab_list() to see if the page is in another tab; then snapshot that tab.
- Always snapshot after navigation or clicks to see the updated page
- Build CSS selectors from attributes shown in the snapshot, never guess
- If a click doesn't work, try waiting first: browser_action(wait, tab=default, target=selector)
- For SPAs, hidden menus/panels appear after clicking their trigger -- snapshot again
- Do not use browser_eval_js to read page content, use browser_snapshot instead
"""

FORM_FILLING_SKILL = """\
---
name: form-filling
description: Fill web forms systematically. Use when completing registration, checkout, or data entry forms.
---

# Form Filling Skill

## Core pattern
1. browser_tab_list() to see open tabs; pick the tab that has the form (or use default).
2. browser_snapshot(tab=...) to discover all form fields and their attributes
3. Build selectors from the snapshot (input[name="email"], select[id="country"])
4. For each field: browser_action(fill, tab=..., target=selector, value=value) for text inputs
5. For dropdowns: browser_action(select, tab=..., target=selector, value=value)
6. For checkboxes: browser_action(click, tab=..., target=selector)
7. Submit: browser_action(click, tab=..., target=submit-button-selector)
8. browser_snapshot(tab=...) to verify success or read error messages

## Tips
- Fill fields in DOM order (top to bottom) to avoid triggering validation early
- For password fields, use browser_wait_for_human to let the operator type
- After submit, check for error messages in the snapshot
- For multi-page forms, snapshot each page before filling
"""

WEB_SCRAPING_SKILL = """\
---
name: web-scraping
description: Extract structured data from web pages. Use when collecting data from tables, lists, or repeated page elements.
---

# Web Scraping Skill

## Quick capture
For saving full page or section content directly to a file:
  browser_save_page(path="output.txt", tab=default, selector=".article-body", wait_for=".article-body")
  browser_save_page(path="page.html", tab=default, format="html")

Use browser_save_page when you need the rendered content saved to disk without
further processing. Use the manual pattern below when you need to transform
or filter the data before saving.

## Manual extraction pattern
1. browser_action(navigate, tab=default, target=URL) to go to the data page
2. browser_snapshot(tab=default) to see the page structure
3. Identify the data pattern (table rows, list items, cards)
4. browser_eval_js(expression, tab=default) to extract structured data via JS
5. Write results to a file with write_file

## JavaScript extraction examples
- Table data: document.querySelectorAll('table tr td').forEach(...)
- List items: [...document.querySelectorAll('.item')].map(el => el.textContent)
- JSON from page: JSON.parse(document.querySelector('script[type="application/ld+json"]').textContent)

## Multi-tab collection
Open multiple tabs to load pages in parallel, then extract from each:
1. browser_tab_new("page1", "https://site.com/article-1")
2. browser_tab_new("page2", "https://site.com/article-2")
3. browser_save_page(path="article1.txt", tab="page1", selector=".content")
4. browser_save_page(path="article2.txt", tab="page2", selector=".content")
5. browser_tab_close("page1") and browser_tab_close("page2")

## Pagination
1. Extract current page data
2. Find "next page" link in snapshot
3. browser_action(click, tab=default, target=next-selector)
4. browser_action(wait, tab=default, target=data-container-selector)
5. Repeat extraction
"""

VISUAL_DEBUGGING_SKILL = """\
---
name: visual-debugging
description: Debug frontend layout issues using screenshots. Use when verifying visual appearance or diagnosing UI problems.
---

# Visual Debugging Skill

## Core pattern
1. browser_action(navigate, tab=default, target=URL) to go to the page
2. browser_screenshot(tab=default) to capture the visual state
3. Describe what you see vs what is expected
4. Use browser_console(tab=default) to check for JS errors
5. Use browser_eval_js(expression, tab=default) to inspect computed styles

## Common checks
- Layout broken: check for CSS errors in console, inspect element dimensions
- Missing content: check network errors in console, verify element exists in snapshot
- Responsive issues: use browser_eval_js("window.innerWidth", tab=default) to check viewport
- Image not loading: check console for 404 errors on image URLs

## Tips
- Take screenshots before and after changes for comparison
- browser_snapshot gives simplified HTML structure, browser_screenshot gives visual
- Use both together for complete debugging
"""


IMPORTANT_BROWSER_SKILLS = [
    ("web-navigation", WEB_NAVIGATION_SKILL),
    ("form-filling", FORM_FILLING_SKILL),
]

GENERIC_BROWSER_SKILLS = [
    ("web-scraping", WEB_SCRAPING_SKILL),
    ("visual-debugging", VISUAL_DEBUGGING_SKILL),
]


def seed_browser_skills(workspace_dir: Path) -> None:
    """Seed built-in browser skills at workspace level.

    Called by BrowserEnvironment on first init. Skills are seeded at
    workspace/.arion/skills/ as a shared base layer.

    When an agent is created with important_skills=["web-navigation"],
    SkillMiddleware copies the matching skills from here into the
    agent's identity_dir/skills/important/ during initial seeding.
    Remaining workspace-level skills serve as a generic fallback.

    After seeding, the agent controls its own skill classification
    by moving folders between important/ and generic/.
    """
    skills_dir = workspace_dir / ".arion" / "skills"

    for name, content in IMPORTANT_BROWSER_SKILLS + GENERIC_BROWSER_SKILLS:
        skill_dir = skills_dir / name
        ensure_directory(skill_dir)
        seed_file(skill_dir / "SKILL.md", content)


def get_browser_skill_names() -> dict[str, list[str]]:
    """Return browser skill names for initial seeding classification.

    Usage:
        names = get_browser_skill_names()
        SkillMiddleware(important_skills=names["important"])

    The important_skills param is seed-time only: it determines initial
    directory placement. The agent can reclassify at runtime by moving
    folders between important/ and generic/.
    """
    return {
        "important": [name for name, _ in IMPORTANT_BROWSER_SKILLS],
        "generic": [name for name, _ in GENERIC_BROWSER_SKILLS],
    }
