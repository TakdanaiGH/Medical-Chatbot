# scraper.py
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; smart-bot/1.0)"
}

def get_threads_from_page(page_url):
    response = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    thread_links = soup.select("div.relative.w-full.h-full > a")

    threads = []
    for a_tag in thread_links:
        thread_url = a_tag['href']
        raw_title = a_tag.get_text(strip=True)
        cleaned_title = raw_title.replace("แพทย์ตอบคำปรึกษาแล้ว", "").strip()
        thread_id = thread_url.split("/")[-1]
        full_url = f"https://www.agnoshealth.com{thread_url}"

        threads.append({
            "id": thread_id,
            "content": cleaned_title,
            "url": full_url
        })

    return threads

def scrape_doctor_answers(thread_url):
    response = requests.get(thread_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    doctor_answers = []
    answers = soup.find_all(
        'li',
        class_="flex flex-col justify-between p-4 w-full h-full rounded-2xl border border-blue-100 bg-white shadow-blue_glow_small"
    )
    for answer in answers:
        label = answer.find('p', class_="font-medium text-white text-sm")
        if label and "คำตอบโดยแพทย์ผู้เชี่ยวชาญ" in label.get_text():
            answer_text = answer.get_text(separator="\n", strip=True)
            doctor_answers.append(answer_text)

    return doctor_answers
