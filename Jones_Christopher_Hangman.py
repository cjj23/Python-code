import random

#Prints a word, with a set of letters star'ed out:
def stars_for_blanks(word, letters_to_star_out):
	stars = ''
	for letter in word:
		if letter in letters_to_star_out:
			stars += '*'
		else:
			stars += letter
	return stars

#Seed the random number generator:
random.seed()
#Set no of lives:
lives = 7

print ("Welcome to Hangman!")
print ("-------------------")

#Open text file of all words to chose from:
f = open("word_list.txt", 'r')
all_words = f.readlines()
f.close()
#Pick a random word to guess from the text file (strip removes the final '\n' from word):
word_to_guess = all_words[random.randrange(len(all_words))].strip()
#This set contains the letters still to guess - start with all letters in word & remove when guessed:
letters_to_guess_set = set(word_to_guess)

#Testing only:
print (word_to_guess)
print (letters_to_guess_set)

#Loop until word is found or all lives are lost:
while len(letters_to_guess_set) > 0:
	print ("\nWord to guess is:")
	print (stars_for_blanks(word_to_guess, letters_to_guess_set))
	print ("Please enter your next guess:")
	letter_guessed = input()
	
	#If letter is in word then no lives lost:
	if letter_guessed in set(word_to_guess):
		#Try will succeed if new letter is guessed (i.e. not in set of letters to be found):
		try: 
			letters_to_guess_set.remove(letter_guessed)
			print ("Letter found! Lives left = " + str(lives))
		except KeyError:
			print ("Letter already found! Lives left = " + str(lives))
	#Letter guessed is not in word. Lose a life:
	else:
		lives -= 1
		print ("Letter not found. Lives left = " + str(lives))
		if lives == 0:
			print ("\nYou lose")
			print ("Word not found  = " + word_to_guess)
			exit()

#If all letters removed from set - ie all letters guessed:
print ("\nCongratulations you win")
print ("Word found  = " + word_to_guess)
