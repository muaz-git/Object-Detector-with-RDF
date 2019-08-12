import rdflib
from rdflib import URIRef, BNode, Literal, Graph
from rdflib.namespace import RDF, FOAF
import requests
import json


# url = 'http://localhost:8080/greeting'
# files = {'file': open('example5.png', 'rb')}
# requests.post(url, files=files)
#
# # URL = "http://localhost:8080/greeting"
# #
# # json_obj = {'name': 'Muaz'}
# # js_str = json.dumps(json_obj)
# # data = {'strr': js_str}
# # headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
# # print(data)
# # r = requests.post(url=URL, data=data, headers=headers)
# # pastebin_url = r.text
# # print("The pastebin URL is:%s" % pastebin_url)
#
# exit()
g = Graph()

img_name = "example.png"
pred = URIRef(img_name + '/pred')
uri = URIRef(img_name)
val = URIRef("1")

g.add((uri, pred, val))

g.serialize(destination='output.xml', format='xml')

# file = open("output.xml", "w")
# file.write(g.serialize(format="turtle"))
