# Python3 program to find largest in an array
# without conditional/bitwise/ternary/ operators
# and without library functions.

# If page found, updates the second chance bit to true
def findAndUpdate(x, arr, second_chance, frames):
	for i in range(frames):
		if arr[i] == x:
			# Mark that the page deserves a second chance
			second_chance[i] = True
			
			# Return 'true', that is there was a hit
			#and so there's no need to replace any page
			return True
	
	# Return 'false' so that a page
	# for replacement is selected
	# as he reuested page doesn't
	# exist in memory
	return False

# Updates the page in memory
# and returns the pointer
def replaceAndUpdate(x, arr, second_chance, frames, pointer):
	while(True):
	
		# We found the page to replace
		if not second_chance[pointer]:
		
			# Replace with new page
			arr[pointer] = x
			
			#Return updated pointer
			return (pointer+1)%frames
		
		# Mark it 'false' as it got one chance
		# and will be replaced next time unless accessed again
		second_chance[pointer] = False
		
		# Pointer is updated in round robin manner
		pointer = (pointer + 1) % frames

def printHitsAndFaults(reference_string, frames):
    # initially we consider
	# frame 0 is to be replaced
	pointer = 0
	
	# number of page faults
	pf = 0
	
	# Create a array to hold page numbers
	arr = [0]*frames
	
	# No pages initially in frame,
	# which is indicated by -1
	for s in range(frames):
		arr[s] = -1
		
	# Create second chance array.
	# Can also be a byte array for optimizing memory
	second_chance = [False]*frames
	
	# Split the string into tokens,
	# that is page numbers, based on space
	Str = reference_string #.split(' ')
	
	# get the length of array
	l = len(Str)
	for i in range(l):
		x = Str[i]
		# Finds if there exists a need to replace
		# any page at all
		if not findAndUpdate(x,arr,second_chance,frames):
			# Selects and updates a victim page
			pointer = replaceAndUpdate(x,arr,second_chance,frames,pointer)
			# Update page faults
			pf += 1
	return pf




