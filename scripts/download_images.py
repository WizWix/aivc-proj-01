import os
import urllib.request
import ssl

def download_image(url, filename):
    print(f"다운로드 중: {filename} ... ({url})")
    try:
        context = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, context=context) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        print(f"다운로드 완료: {filename}")
    except Exception as e:
        print(f"다운로드 실패: {e}")

# Wikipedia links for test images
# Using random Wikipedia Commons public domain / CC-BY celeb images
test_urls = {
    "tom_cruise_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/33/Tom_Cruise_by_Gage_Skidmore_2.jpg",
    "tom_cruise_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/71/Tom_Cruise_avp_2014_4.jpg",
    "brad_pitt.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg"
}

os.makedirs("images", exist_ok=True)
for name, url in test_urls.items():
    download_image(url, os.path.join("images", name))
