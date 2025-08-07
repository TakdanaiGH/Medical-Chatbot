# main.py
import os
import json
import time
from scraper import get_threads_from_page, scrape_doctor_answers
from dotenv import load_dotenv

load_dotenv()

json_path = os.getenv("RAG_JSON", "agnos_threads.json")

base_url = "https://www.agnoshealth.com/forums/search?page="
all_threads = []

# Load existing data
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            all_threads = json.load(f)
        except json.JSONDecodeError:
            all_threads = []

existing_ids = set(thread["id"] for thread in all_threads)

print("🔄 Thread watcher started. Scanning all pages every 5 minutes...\n")

while True:
    print("🔍 Starting full scan...")
    page = 1
    new_threads_count = 0

    while True:
        url = base_url + str(page)
        print(f"📄 Scraping page {page}")
        threads = get_threads_from_page(url)

        if not threads:
            print("❌ No threads found. Reached end.")
            break

        page_new_threads = 0

        for thread in threads:
            if thread["id"] in existing_ids:
                continue

            # Add doctor answers only (no main post)
            doctor_answers = scrape_doctor_answers(thread["url"])
            thread["doctor_answers"] = doctor_answers
            thread["page"] = page

            all_threads.append(thread)
            existing_ids.add(thread["id"])
            page_new_threads += 1
            new_threads_count += 1

        if page_new_threads > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_threads, f, indent=4, ensure_ascii=False)
            print(f"✅ Page {page}: {page_new_threads} new thread(s) saved.")
        else:
            print(f"🆗 Page {page}: No new threads.")

        page += 1

    print(f"\n🔁 Scan complete. {new_threads_count} new thread(s) added.")
    print("⏳ Waiting 5 minutes before next cycle...\n")
    time.sleep(300)
