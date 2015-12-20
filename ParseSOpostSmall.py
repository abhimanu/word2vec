mport xml.etree.cElementTree as etree

xml_file = open('SOposts1000.xml','r')
tree = etree.iterparse(xml_file)
for events, row in tree:
	if 'Title' in row.attrib:
		print row.attrib['Title']
