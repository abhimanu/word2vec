import xml.etree.cElementTree as etree
import sys

xml_file = open(sys.argv[1],'r')
tree = etree.iterparse(xml_file)
for events, row in tree:
	if 'Title' in row.attrib:
		print row.attrib['Title'].encode('utf-8')
