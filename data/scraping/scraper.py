import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base configuration
BASE_URL = "https://www.grabcraft.com"
CATEGORIES = {
    "buildings": "https://www.grabcraft.com/minecraft/buildings",
    "military": "https://www.grabcraft.com/minecraft/military-buildings",
    "fantasy": "https://www.grabcraft.com/minecraft/fantasy-buildings"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_grabcraft():
    for tag, cat_url in CATEGORIES.items():
        print(f"--- Scraping Category: {tag} ---")
        
        # Create directory for the category
        os.makedirs(f"data/{tag}", exist_ok=True)
        
        response = requests.get(cat_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to load {cat_url}")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all build links (usually inside specific div classes)
        # Note: GrabCraft uses <a> tags with specific titles or classes for builds
        links = soup.select('a[href*="/minecraft/"]')
        
        processed_links = set()
        
        for link in links:
            build_url = urljoin(BASE_URL, link['href'])
            
            # Filter to ensure we are hitting a build page, not a sub-category
            if "/minecraft/" in build_url and build_url not in processed_links:
                if build_url == cat_url: continue
                
                print(f"Found build: {build_url}")
                save_build_info(build_url, tag)
                processed_links.add(build_url)
                
                # Ethical scraping: delay to avoid IP blocks
                time.sleep(1)

def save_build_info(url, tag):
    """
    Saves the metadata of the build. 
    To get the actual voxel data, you'd need to parse their 'renderObject' JS.
    """
    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        name = soup.find('h1').text.strip() if soup.find('h1') else "unknown"
        clean_name = "".join(x for x in name if x.isalnum())
        
        file_path = f"data/{tag}/{clean_name}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Name: {name}\n")
            f.write(f"Category: {tag}\n")
            f.write(f"URL: {url}\n")
            # You can add logic here to scrape block counts or descriptions
            
    except Exception as e:
        print(f"Error saving {url}: {e}")

if __name__ == "__main__":
    scrape_grabcraft()