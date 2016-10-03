from distutils.core import setup


setup(
  name = 'pyfmm',
  packages = ['pyfmm'], # this must be the same as the name above
  version = '0.1',
  description = 'Python module implementing the Fast Marching Method.',
  #long_description=open('README.md', 'rt').read(),
  author = 'Vegard Kvernelv',
  author_email = 'vkvernelv@gmail.com',
  url = 'https://github.com/vegardkv/pyfmm', # use the URL to the github repo
  download_url = 'https://github.com/vegardkv/pyfmm/tarball/0.1', # I'll explain this in a second
  keywords = ['fast marching method', 'eikonal', 'pyfmm', 'distance'], # arbitrary keywords
  classifiers = [],
)