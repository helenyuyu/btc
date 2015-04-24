f = open('reduced_txTripletsCounts.txt', 'r')
f_out = open('btc.dat', 'w')
prev_doc = 0
triplets = f.readlines()
line = ""
count = 0
for l in triplets:
	triplet = l.split()
	current = int(triplet[0])
	if (current != prev_doc):
		f_out.write(str(count) + line + "\n")
		prev_doc += 1
		while (prev_doc != current):
			f_out.write("0\n") # for senders who do not send anything
			prev_doc+=1
		line = ""
		count = 0
	line += " " + triplet[1] + ":" + triplet[2]
	count += int(triplet[2])

f_out.write(str(count) + " " + line)

f.close()
f_out.close()

