__author__ = 'fangy'

import sys
from xml.etree.cElementTree import iterparse

def processItem(item):
    """
    Process a review.
    Implement custom code here. Use 'item.find('tagname').text' to access the properties of a review.
    """
    attr_names = ['category', 'polarity', 'rating', 'summary',
                  'text']
    attr_value = {}
    for attr_name in attr_names:
        node = item.find(attr_name)
        if node is None: continue
        attr_value[attr_name] = node.text

    return attr_value


def process_cn(item_xml_file):



    for event, elem in iterparse(item_xml_file):
        if elem.tag == 'item':
            attr_value = processItem(elem)
            print attr_value['text']
            elem.clear()

if __name__ == "__main__":

    fname = sys.argv[1]
    if not fname.endswith(".xml"):
        print >> sys.stderr, 'file should ends with "xml"'
    else:
        fd = open(fname)
    process_cn(fd)
