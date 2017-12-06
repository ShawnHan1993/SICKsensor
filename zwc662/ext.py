import xml.etree.ElementTree as ET
import sys

def extract(input_path, _LFT = False, _NotLFT = False):


 file_name = open(input_path, 'r')
 if _LFT is True:
	output_path = "LFT_"
 elif _NotLFT is True:
	output_path = "NotLFT_"
 else:
	output_path = "objectdata_"

 output_path = output_path + input_path.split('.')[1] + ".xml"
 data = ET.Element(tag = 'data', attrib = {'filename': input_path})

 lines = file_name.readlines()
 for index in range(0, len(lines)):
#for index in range(0, 1):
	line = lines[index]
	line = "<data>" + line + "</data>"
	##print line
	root = ET.fromstring(line)

	flag = False
	token = None

	for objectdata in root.iter('objectdata'):
		if objectdata.tag == 'objectdata':
			flag = True
			break

	if flag is True:
		for tokenid in root.iter('tokenid'):
			#print tokenid.text
			token = tokenid.text
	else:
		continue
	flag = False

	objectdata = ET.SubElement(data, 'objectivedata')

	objectdata.attrib = {'tokenid': token} 

	#object size
	for volumetric in root.iter('volumetric'):
		for size in volumetric.iter('size'): 
			if size.tag == 'size':
				flag = True
				#print size.attrib['unit']
				unit = size.attrib['unit']
				sizedata = ET.Element(tag = 'size', attrib = {'unit':unit})

				#print size.attrib['ohe']
				ohe = ET.SubElement(sizedata, 'ohe')
				ohe.text = size.attrib['ohe']

				#print size.attrib['owi']
				owi = ET.SubElement(sizedata, 'owi')
				owi.text = size.attrib['owi']

				#print size.attrib['ole']
				ole = ET.SubElement(sizedata, 'ole')
				ole.text = size.attrib['ole']

				objectdata.append(sizedata)
			break
		break	
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#object weight
	for scaledata in root.iter('scaledata'):
		for owe in scaledata.iter('owe'):
			if owe.tag == 'owe':
				flag = True
				#print owe.get('unit')
				unit = owe.get('unit')
				weightdata = ET.Element(tag = 'weight', attrib = {'unit':unit})

				#print owe.find('value').text
				weight = ET.SubElement(weightdata, 'owe')
				weight.text = owe.find('value').text

				objectdata.append(weightdata)
			break
		break
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#gap
	for general in root.iter('general'):
		for oga in general.iter('oga'):
			if oga.tag == 'oga':
				flag = True
				#print oga.get('unit')
				unit = oga.get('unit')
				gapdata = ET.Element(tag = 'gap', attrib = {'unit':unit})

				#print oga.find('value').text
				gap = ET.SubElement(gapdata, 'oga')
				gap.text = oga.find('value').text

				objectdata.append(gapdata)
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#box volume
	for volumetric in root.iter('volumetric'):
		for obv in volumetric.iter('obv'):
			if obv.tag == 'obv':
				flag = True
				#print obv.get('unit')
				unit = obv.get('unit')
				volumedata = ET.Element(tag = 'volume', attrib = {'unit': unit})
				#print obv.find('value').text
				volume = ET.SubElement(volumedata, 'obv')
				volume.text = obv.find('value').text
				
				objectdata.append(volumedata)
	
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#object orientation
		for oa in volumetric.iter('oa'):
			if oa.tag == 'oa':
				flag = True
				#print oa.get('unit')
				unit = oa.get('unit')
				orientationdata = ET.Element(tag = 'orientation', attrib = {'unit': unit})
				#print oa.find('value').text
				orientation = ET.SubElement(orientationdata, 'oa')
				orientation.text = oa.find('value').text
				
				objectdata.append(orientationdata)
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#object speed
		for otve in volumetric.iter('otve'):
			if otve.tag == 'otve':
				flag = True
				#print otve.get('unit')
				unit = otve.get('unit')
				speeddata = ET.Element(tag = 'speed', attrib = {'unit': unit})
				#print otve.find('value').text
				speed = ET.SubElement(speeddata, 'otve')
				speed.text = otve.find('value').text
				
				objectdata.append(speeddata)

	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		flag = False
	#conveyor speed 
	for sorterstate in root.iter('sorterstate'):
		for speed in sorterstate.iter('speed'):
			if speed.tag == 'speed':
				flag = True
				#print speed.get('unit')
				unit = speed.get('unit')
				convoyer_speeddata = ET.Element(tag = 'conveyor_speed', attrib = {'unit': unit})
				#print speed.find('value').text
				cve = ET.SubElement(convoyer_speeddata, 'cve')
				cve.text = speed.find('value').text
			
				objectdata.append(convoyer_speeddata)

	if flag is False:
		data.remove(objectdata)
		continue	

	flag = False

	condition = ET.Element(tag = 'condition')
	#PDFw9612 = ET.SubElement(condition, 'PDFw9612')
	#PDFNoRead = ET.SubElement(condition, 'PDFNoRead')
	TooBig = ET.SubElement(condition, 'TooBig')
	TooBig.text = '0'

	NoRead = ET.SubElement(condition, 'NoRead')
	NoRead.text = '0'
			
	NotLFT = ET.SubElement(condition, 'NotLFT')
	NotLFT.text = '0'

	MultiRead = ET.SubElement(condition, 'MultiRead')
	MultiRead.text = '0'

	Irreg = ET.SubElement(condition, 'Irreg')
	Irreg.text = '0'
 
	TooSmall = ET.SubElement(condition, 'TooSmall')
	TooSmall.text = '0'

	LFT = ET.SubElement(condition, 'LFT')
	LFT.text = '0'	
			
	for condition_data in root.iter('condition'):
		if True or condition_data.tag == 'condition':
			for condition_str in condition_data.text.split(','):
				if condition_str == 'TooBig':
					TooBig.text = '1'
				elif condition_str == 'NoRead':
					NoRead.text = '1'
				elif condition_str == 'NotLFT':	
					if _NotLFT is True:
						NotLFT.text = '1'
						flag = True
				elif condition_str == 'MultiRead':
					MultiRead.text = '1'
				elif condition_str == 'Irreg':
					Irreg.text = '1'
				elif condition_str == 'TooSmall':
					TooSmall.text = '1'
				elif condition_str == 'LFT':
					if _LFT is True:
						LFT.text = '1'
						flag = True
		
						
	if flag is False:
		data.remove(objectdata)
		continue	
	else:
		objectdata.append(condition)
		flag = False

 tree = ET.ElementTree(None)
 tree._setroot(data)
 #ET.dump(tree)
 tree.write(output_path)


if __name__ == "__main__":
	input_paths = sys.argv[1:]
	for input_path in input_paths:
		extract(input_path, _LFT = True, _NotLFT = False)
		extract(input_path, _LFT = False, _NotLFT = True)
