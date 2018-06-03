"""Script to download images from a text file with a list of URLs.
Text file should have one URL per line.
"""
import argparse
import requests
import os

parser = argparse.ArgumentParser(description='Image Data Downloader')
parser.add_argument('--url_file', default=None,
                    type=str, help='File path of URLs text file')
parser.add_argument('--save_folder', default='image_data', type=str,
                    help='Dir to save downloaded images')
args = parser.parse_args()

def main():
    save_folder = os.path.join(os.getcwd(), args.save_folder)
    if not os.path.exists(args.save_folder):
        os.makedirs(save_folder)
    with open(args.url_file, 'r') as f:
        for line in f:
            url = line.rstrip()
            print(url)
            r = requests.get(url, allow_redirects=True)
            open(os.path.join(save_folder, os.path.basename(url.split('/')[-1])), 'wb').write(r.content)

if __name__ == '__main__':
    main()