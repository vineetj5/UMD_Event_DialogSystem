# import re
# import time
# import json
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from datetime import datetime, date, timedelta

# # --- Optional progress bar (tqdm). Falls back to simple prints if not installed.
# try:
#     from tqdm import tqdm
#     def pbar(iterable, desc="", unit="it"):
#         return tqdm(iterable, desc=desc, unit=unit, leave=False)
# except Exception:
#     def pbar(iterable, desc="", unit="it"):
#         total = len(iterable) if hasattr(iterable, "__len__") else None
#         count = 0
#         print(f"{desc} ...")
#         for x in iterable:
#             yield x
#             count += 1
#             if total:
#                 pct = int((count / total) * 100)
#                 if pct % 10 == 0:
#                     print(f"  {desc}: {pct}%")
#         print(f"{desc}: done")

# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) "
#         "Gecko/20100101 Firefox/120.0"
#     ),
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#     "Accept-Language": "en-US,en;q=0.5",
#     "Referer": "https://calendar.umd.edu/",
#     "Connection": "keep-alive",
# }
# SESSION = requests.Session()
# SESSION.headers.update(HEADERS)

# BASE_DAY_URL = "https://calendar.umd.edu/events/{yyyy}/{mm}/{dd}"
# OUTFILE = "umd_calendar_2025-10-01_to_2025-10-31.json"
# REQUEST_DELAY_SEC = 0.25  # be polite


# def clean(txt: str) -> str:
#     return " ".join((txt or "").replace("\xa0", " ").split())


# def fetch_soup(url: str) -> BeautifulSoup:
#     r = SESSION.get(url, timeout=30)
#     r.raise_for_status()
#     return BeautifulSoup(r.text, "html.parser")


# def extract_umd_description(soup: BeautifulSoup) -> str:
#     """
#     Description lives under <umd-event-description> on the detail page.
#     """
#     tag = soup.find("umd-event-description")
#     if not tag:
#         return "N/A"
#     ps = tag.find_all("p")
#     if ps:
#         paras = [clean(p.get_text(" ", strip=True)) for p in ps if clean(p.get_text(" ", strip=True))]
#         if paras:
#             return "\n\n".join(paras)
#     txt = clean(tag.get_text(" ", strip=True))
#     return txt if txt else "N/A"


# def extract_location(soup: BeautifulSoup) -> str:
#     """
#     Find location on detail page using <umd-event-location> and common patterns.
#     """
#     # 1) Preferred: <umd-event-location> (seen in your screenshot)
#     loc_tag = soup.find("umd-event-location")
#     if loc_tag:
#         txt = clean(loc_tag.get_text(" ", strip=True))
#         if txt:
#             return txt

#     # 2) dl/dt/dd with labels like Where/Location
#     for dt_tag in soup.select("dt"):
#         if clean(dt_tag.get_text()).rstrip(":").lower() in {"where", "location", "place", "venue"}:
#             dd = dt_tag.find_next_sibling("dd")
#             if dd:
#                 txt = clean(dd.get_text(" ", strip=True))
#                 if txt:
#                     return txt

#     # 3) strong/b labels
#     for strong in soup.select("strong, b"):
#         label = clean(strong.get_text()).rstrip(":").lower()
#         if label in {"where", "location", "place", "venue"}:
#             txt = clean(strong.parent.get_text(" ", strip=True).replace(strong.get_text(), "", 1))
#             if txt:
#                 return txt

#     # 4) Generic classes
#     for sel in [".event__location", ".event-location", ".field--name-field-location", '[class*="location"]']:
#         el = soup.select_one(sel)
#         if el:
#             txt = clean(el.get_text(" ", strip=True))
#             if txt:
#                 return txt

#     return "N/A"


# def _parse_dt(val: str):
#     """Parse 'YYYY-MM-DD HH:MM:SS' or ISO 'YYYY-MM-DDTHH:MM:SS±ZZ:ZZ'."""
#     val = val.strip()
#     for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
#         try:
#             return datetime.strptime(val, fmt)
#         except Exception:
#             continue
#     return None


# def extract_date_time_from_flags(soup: BeautifulSoup):
#     """
#     On the detail page:

#     <umd-calendar-flags>
#       <ul>
#         <li> ... <time datetime="2025-10-02T08:00:00-04:00">Thu Oct 02</time> ... </li>
#         <li> ... <time datetime="2025-10-02T12:00:00-04:00">12:00pm</time>
#                   <time datetime="2025-10-02T13:00:00-04:00">1:00pm</time> ... </li>
#       </ul>

#     We want:
#       date = ISO date or visible date text
#       time = "12:00pm - 1:00pm"
#     """
#     flags = soup.find("umd-calendar-flags")
#     if not flags:
#         return "N/A", "N/A"

#     lis = flags.find_all("li")
#     date_str, time_str = "N/A", "N/A"

#     # --- DATE from first <li> ---
#     if lis:
#         t_date = lis[0].find("time")
#         if t_date:
#             dt_attr = t_date.get("datetime")
#             if dt_attr:
#                 dt = _parse_dt(dt_attr)
#                 if dt:
#                     date_str = dt.date().isoformat()
#             if date_str == "N/A":
#                 # fallback to visible text e.g. "Thu Oct 02"
#                 txt = clean(t_date.get_text(" ", strip=True))
#                 if txt:
#                     date_str = txt

#     # --- TIME from second <li> (time range) ---
#     if len(lis) > 1:
#         time_tags = lis[1].find_all("time")
#         if time_tags:
#             texts = [clean(t.get_text(" ", strip=True)) for t in time_tags if clean(t.get_text(" ", strip=True))]
#             if len(texts) == 1:
#                 time_str = texts[0]
#             elif len(texts) >= 2:
#                 time_str = f"{texts[0]} - {texts[1]}"

#             # If no visible text, try datetime attributes
#             if time_str == "N/A":
#                 parsed = []
#                 for t in time_tags:
#                     t_attr = t.get("datetime")
#                     tdt = _parse_dt(t_attr) if t_attr else None
#                     if tdt:
#                         parsed.append(tdt.strftime("%-I:%M%p").lower())
#                 if parsed:
#                     if len(parsed) == 1:
#                         time_str = parsed[0]
#                     else:
#                         time_str = f"{parsed[0]} - {parsed[1]}"

#     return date_str, time_str


# def day_url(d: date) -> str:
#     return BASE_DAY_URL.format(yyyy=d.year, mm=f"{d.month:02d}", dd=f"{d.day:02d}")


# def scrape_day(d: date):
#     """
#     For a single day:
#       - Load the day listing page
#       - For each event <h2 class="event-title heading-san-four">, grab the <a> href
#       - Go to that event link (detail page)
#       - Extract title, date, time, location, description
#     """
#     url = day_url(d)
#     day_soup = fetch_soup(url)
#     time.sleep(REQUEST_DELAY_SEC)

#     h2s = day_soup.select("h2.event-title.heading-san-four")
#     for h2 in pbar(h2s, desc=f"{d.isoformat()} events", unit="evt"):
#         a = h2.find("a", href=True)
#         if not a:
#             continue

#         detail_url = urljoin(url, a["href"])

#         try:
#             detail = fetch_soup(detail_url)
#         except Exception as e:
#             print(f"[WARN] Failed {detail_url}: {e}")
#             continue

#         # Title from detail page (prefer h1)
#         title_tag = detail.find("h1")
#         if title_tag:
#             title = clean(title_tag.get_text(" ", strip=True))
#         else:
#             title = clean(a.get_text(" ", strip=True))

#         desc = extract_umd_description(detail)
#         loc = extract_location(detail)
#         ev_date, ev_time = extract_date_time_from_flags(detail)

#         yield {
#             "event": title or "N/A",
#             "date": ev_date or "N/A",
#             "time": ev_time or "N/A",
#             "url": detail_url or "N/A",
#             "location": loc or "N/A",
#             "description": desc or "N/A",
#         }

#         time.sleep(REQUEST_DELAY_SEC)


# def main():
#     start = date(2025, 10, 1)
#     end   = date(2025, 10, 2)

#     seen_urls = set()
#     collected = []

#     # Build list of dates first so we can show a days progress bar
#     days = []
#     cur = start
#     while cur <= end:
#         days.append(cur)
#         cur += timedelta(days=1)

#     for d in pbar(days, desc="Days", unit="day"):
#         try:
#             for ev in scrape_day(d):
#                 if ev["url"] in seen_urls:
#                     continue
#                 seen_urls.add(ev["url"])
#                 collected.append(ev)
#         except Exception as e:
#             print(f"[WARN] Day {d.isoformat()} failed: {e}")

#     if not collected:
#         print("No events collected for the range.")
#         return

#     # Save all events as a JSON list
#     with open(OUTFILE, "w", encoding="utf-8") as f:
#         json.dump(collected, f, ensure_ascii=False, indent=2)

#     print(f"✅ Saved {len(collected)} events to '{OUTFILE}' in JSON format")


# if __name__ == "__main__":
#     main()



import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, date, timedelta

# --- Optional progress bar (tqdm). Falls back to simple prints if not installed.
try:
    from tqdm import tqdm
    def pbar(iterable, desc="", unit="it"):
        return tqdm(iterable, desc=desc, unit=unit, leave=False)
except Exception:
    def pbar(iterable, desc="", unit="it"):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        count = 0
        print(f"{desc} ...")
        for x in iterable:
            yield x
            count += 1
            if total:
                pct = int((count / total) * 100)
                if pct % 10 == 0:
                    print(f"  {desc}: {pct}%")
        print(f"{desc}: done")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://calendar.umd.edu/",
    "Connection": "keep-alive",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

BASE_DAY_URL = "https://calendar.umd.edu/events/{yyyy}/{mm}/{dd}"
OUTFILE = "umd_calendar_2025-10-01_to_2025-10-31.json"
REQUEST_DELAY_SEC = 0.25  # be polite


def clean(txt: str) -> str:
    return " ".join((txt or "").replace("\xa0", " ").split())


def fetch_soup(url: str) -> BeautifulSoup:
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_umd_description(soup: BeautifulSoup) -> str:
    """
    Description lives under <umd-event-description> on the detail page.
    """
    tag = soup.find("umd-event-description")
    if not tag:
        return "N/A"
    ps = tag.find_all("p")
    if ps:
        paras = [clean(p.get_text(" ", strip=True)) for p in ps if clean(p.get_text(" ", strip=True))]
        if paras:
            return "\n\n".join(paras)
    txt = clean(tag.get_text(" ", strip=True))
    return txt if txt else "N/A"


def extract_location(soup: BeautifulSoup) -> str:
    """
    Find location on detail page using <umd-event-location> and common patterns.
    """
    # 1) Preferred: <umd-event-location>
    loc_tag = soup.find("umd-event-location")
    if loc_tag:
        txt = clean(loc_tag.get_text(" ", strip=True))
        if txt:
            return txt

    # 2) dl/dt/dd with labels like Where/Location
    for dt_tag in soup.select("dt"):
        if clean(dt_tag.get_text()).rstrip(":").lower() in {"where", "location", "place", "venue"}:
            dd = dt_tag.find_next_sibling("dd")
            if dd:
                txt = clean(dd.get_text(" ", strip=True))
                if txt:
                    return txt

    # 3) strong/b labels
    for strong in soup.select("strong, b"):
        label = clean(strong.get_text()).rstrip(":").lower()
        if label in {"where", "location", "place", "venue"}:
            txt = clean(strong.parent.get_text(" ", strip=True).replace(strong.get_text(), "", 1))
            if txt:
                return txt

    # 4) Generic classes
    for sel in [".event__location", ".event-location", ".field--name-field-location", '[class*="location"]']:
        el = soup.select_one(sel)
        if el:
            txt = clean(el.get_text(" ", strip=True))
            if txt:
                return txt

    return "N/A"


def _parse_dt(val: str):
    """Parse 'YYYY-MM-DD HH:MM:SS' or ISO 'YYYY-MM-DDTHH:MM:SS±ZZ:ZZ'."""
    val = val.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(val, fmt)
        except Exception:
            continue
    return None


def extract_date_time_from_flags(soup: BeautifulSoup):
    """
    Extract date and time from <umd-calendar-flags>.

    For range dates like:
      "Mon Sep 15 – Wed Oct 01" (often with a hidden "To"),
    we keep the whole human-readable range but normalize it to use only "-",
    e.g. "Mon Sep 15 - Wed Oct 01".
    """
    flags = soup.find("umd-calendar-flags")
    if not flags:
        return "N/A", "N/A"

    lis = flags.find_all("li")
    date_str, time_str = "N/A", "N/A"

    # --- DATE from first <li> ---
    if lis:
        date_li = lis[0]
        date_times = date_li.find_all("time")

        if len(date_times) > 1:
            # Date RANGE case: keep full text, normalize it
            txt_raw = date_li.get_text(" ", strip=True)

            # Replace en/em dash with regular dash
            txt = txt_raw.replace("–", "-").replace("—", "-")

            # Remove the word "to"/"To"/"TO" etc.
            txt = re.sub(r"\bto\b", "", txt, flags=re.IGNORECASE)

            # Normalize spaces around dash to " - "
            txt = re.sub(r"\s*-\s*", " - ", txt)

            # Final whitespace cleanup
            txt = clean(txt)

            if txt:
                date_str = txt
        else:
            # Single date: as before, prefer datetime attr → ISO, else visible text
            t_date = date_li.find("time")
            if t_date:
                dt_attr = t_date.get("datetime")
                if dt_attr:
                    dt = _parse_dt(dt_attr)
                    if dt:
                        date_str = dt.date().isoformat()
                if date_str == "N/A":
                    txt = clean(t_date.get_text(" ", strip=True))
                    if txt:
                        date_str = txt

    # --- TIME from second <li> (time range) ---
    if len(lis) > 1:
        time_li = lis[1]
        time_tags = time_li.find_all("time")
        if time_tags:
            texts = [clean(t.get_text(" ", strip=True)) for t in time_tags if clean(t.get_text(" ", strip=True))]
            if len(texts) == 1:
                time_str = texts[0]
            elif len(texts) >= 2:
                time_str = f"{texts[0]} - {texts[1]}"

            # If still empty, try datetime attributes
            if time_str == "N/A":
                parsed = []
                for t in time_tags:
                    t_attr = t.get("datetime")
                    tdt = _parse_dt(t_attr) if t_attr else None
                    if tdt:
                        parsed.append(tdt.strftime("%-I:%M%p").lower())
                if parsed:
                    if len(parsed) == 1:
                        time_str = parsed[0]
                    else:
                        time_str = f"{parsed[0]} - {parsed[1]}"

    return date_str, time_str


def day_url(d: date) -> str:
    return BASE_DAY_URL.format(yyyy=d.year, mm=f"{d.month:02d}", dd=f"{d.day:02d}")


def scrape_day(d: date):
    """
    For a single day:
      - Load the day listing page
      - For each event <h2 class="event-title heading-san-four">, grab the <a> href
      - Go to that event link (detail page)
      - Extract title, date, time, location, description
    """
    url = day_url(d)
    day_soup = fetch_soup(url)
    time.sleep(REQUEST_DELAY_SEC)

    h2s = day_soup.select("h2.event-title.heading-san-four")
    for h2 in pbar(h2s, desc=f"{d.isoformat()} events", unit="evt"):
        a = h2.find("a", href=True)
        if not a:
            continue

        detail_url = urljoin(url, a["href"])

        try:
            detail = fetch_soup(detail_url)
        except Exception as e:
            print(f"[WARN] Failed {detail_url}: {e}")
            continue

        # Title from detail page (prefer h1)
        title_tag = detail.find("h1")
        if title_tag:
            title = clean(title_tag.get_text(" ", strip=True))
        else:
            title = clean(a.get_text(" ", strip=True))

        desc = extract_umd_description(detail)
        loc = extract_location(detail)
        ev_date, ev_time = extract_date_time_from_flags(detail)

        yield {
            "event": title or "N/A",
            "date": ev_date or "N/A",
            "time": ev_time or "N/A",
            "url": detail_url or "N/A",
            "location": loc or "N/A",
            "description": desc or "N/A",
        }

        time.sleep(REQUEST_DELAY_SEC)


def main():
    start = date(2025, 12, 1)
    end   = date(2026, 1, 31)

    seen_urls = set()
    collected = []

    # Build list of dates first so we can show a days progress bar
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    for d in pbar(days, desc="Days", unit="day"):
        try:
            for ev in scrape_day(d):
                if ev["url"] in seen_urls:
                    continue
                seen_urls.add(ev["url"])
                collected.append(ev)
        except Exception as e:
            print(f"[WARN] Day {d.isoformat()} failed: {e}")

    if not collected:
        print("No events collected for the range.")
        return

    # Save all events as a JSON list
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(collected)} events to '{OUTFILE}' in JSON format")


if __name__ == "__main__":
    main()
