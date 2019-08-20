import glob,csv,pdb

path = '/cvhci/data/PersonREid/FacialCosmeticContent/*/*/*'

files = glob.glob(path)

urls = {}

for i in files:
	if 'AnyMakeup' in i:
		anymkup = [i.split(i.split('/')[-1][0:12])[1].split(i[-16:])[0]]
		nomkup  = []
		strmkup = []

	elif 'StrongMakeup' in i:
		strmkup = [i.split(i.split('/')[-1][0:12])[1].split(i[-16:])[0]]
		nomkup  = []
		anymkup = []

	elif 'NoMakeup' in i:
		nomkup = [i.split(i.split('/')[-1][0:12])[1].split(i[-16:])[0]]
		anymkup = []
		strmkup = []


	if i.split('/')[-1][0:11] not in list(urls.keys()):
		urls[i.split('/')[-1][0:11]] = [nomkup,anymkup,strmkup]
	else:
		urls[i.split('/')[-1][0:11]] = [urls[i.split('/')[-1][0:11]][0]+nomkup,urls[i.split('/')[-1][0:11]][1]+anymkup,urls[i.split('/')[-1][0:11]][2]+strmkup]
pdb.set_trace()

with open('yt_frames_.csv','w') as file:
	w = csv.writer(file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in urls.keys():
		w.writerow([i,';'.join(urls[i][0]),';'.join(urls[i][1]),';'.join(urls[i][2])])