#!/usr/bin/python
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

#Problem:
# An image of an animal has been broken into many narrow vertical stripe images.
# The stripe images are on the internet somewhere, each with its own url.
# The urls are hidden in a web server log file.
# Your mission is to find the urls and download all image stripes to re-create the original image.
###################################################################################################


import os
import re
import sys
import  urllib.request
import pathlib

"""Logpuzzle exercise
Given an apache logfile, find the puzzle urls and download the images.

Here's what a puzzle url looks like:
10.254.254.28 - - [06/Aug/2007:00:13:48 -0700] "GET /~foo/puzzle-bar-aaab.jpg HTTP/1.0" 302 528 "-" "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6"
"""


def read_urls(filename):
  """Returns a list of the puzzle urls from the given log file,
  extracting the hostname from the filename itself.
  Screens out duplicate urls and returns the urls sorted into
  increasing order."""
  # +++your code here+++
  cwd = os.getcwd()
  #print(cwd)

  hostIndex = filename.index('_')
  host = filename[hostIndex + 1:]
  #print("Host", hostIndex, " ", host)

  f = open(filename)
  imageList = []
  for line in f:
    match = re.search(r'GET\W+(.+puzzle.+\.jpg)',line)
    if match:
      #print(match.group(0), ",", match.group(1))
      imageLink = 'http://' + host + match.group(0).removeprefix('GET').strip()
      if imageLink not in imageList:
        imageList.append(imageLink)
        #print(imageList[-1])
  #print(len(imageList))

  imageList.sort()
  print("Sorted: ", imageList)
  return imageList


def download_images(img_urls, dest_dir):
  """Given the urls already in the correct order, downloads
  each image into the given directory.
  Gives the images local filenames img0, img1, and so on.
  Creates an index.html in the directory
  with an img tag to show each local image file.
  Creates the directory if necessary.
  """
  # +++your code here+++

  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  index = open(os.path.join(dest_dir, 'index.html'), 'w')
  index.write('<html><body>\n')

  i = 0
  for url in img_urls:
    try:
      print ("started ", url)

      outFilename = 'image%d' % i
      urllib.request.urlretrieve(url, os.path.join(dest_dir, outFilename))

      index.write('<img src="%s">' % outFilename)
      i += 1

      """
      info =ufile.info()
      print(ufile.info().get_content_type())
      imageContent = ufile.read()
      """

    except IOError:
      print('problem reading url:', url)
  index.close()
  return

def main():

  """
  args = sys.argv[1:]

  if not args:
    print 'usage: [--todir dir] logfile '
    sys.exit(1)

  todir = ''
  if args[0] == '--todir':
    todir = args[1]
    del args[0:2]


  img_urls = read_urls(args[0])

  if todir:
    download_images(img_urls, todir)
  else:
    print '\n'.join(img_urls)
 """

  img_urls = read_urls('animal_code.google.com')
  todir = 'out'
  print ("Calling download")
  download_images(img_urls, todir)
if __name__ == '__main__':
  main()
