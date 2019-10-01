import urllib.request
import os.path


def download_file_and_save(url, file_name):
    """Download the file from `url` and save it locally under `file_name`:

    Args:
        url (string): The url to download the file from
        file_name (string): The name of the file
    """
    if not os.path.exists('./' + file_name):
        print('Downloading file from {}...'.format(url))
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()  # a `bytes` object
            print('Saving file as {}...'.format(file_name))
            out_file.write(data)


if __name__ == '__main__':
    URL = 'http://www.gutenberg.org/cache/epub/10/pg10.txt'
    FILENAME = 'bible.txt'

    download_file_and_save(URL, FILENAME)

    # Open the file with read only permit
    with open('bible.txt', 'r') as f:
        lines = f.readlines()
        print('{} lines'.format(len(lines)))

        print('Removing empty lines...')
        lines[:] = [line.lower().replace('\n', '') for line in lines if line != '\n']

        print('{} lines'.format(len(lines)))
        print(lines[:10])
