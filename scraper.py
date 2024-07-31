import requests
from bs4 import BeautifulSoup
import os

# Define the URL to scrape
url = 'https://www.casamance.com/en/'

# Send a GET request to the URL
response = requests.get(url)

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image tags
img_tags = soup.find_all('img')

# Create a directory to save the images
if not os.path.exists('images'):
    os.makedirs('images')

# Loop through each image tag and save the image
for img in img_tags:
    img_url = img.get('src')
    if img_url:
        # Handle relative URLs
        if not img_url.startswith('http'):
            img_url = url + img_url
        
        # Get the image file name
        img_name = os.path.basename(img_url)
        
        # Send a GET request to fetch the image
        img_response = requests.get(img_url)
        
        # Save the image
        with open(os.path.join('images', img_name), 'wb') as img_file:
            img_file.write(img_response.content)
            print(f'Saved {img_name}')

print('Image scraping completed.')
