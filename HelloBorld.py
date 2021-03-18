index = {
          "brutus" : [1,2,4,11,31,45,173,174],
          "caesar" : [1,2,4,6,16,57,132],
          "calpurnia" : [2,31,54,101]
        }

print("The index is: ")
print(index)

def post_list(term):
  return index[term]

print("For example, the is the posting list for 'brutus': ")
print(post_list("brutus"))
print()

def intersect(p1, p2):
  answer = []
  i = j = 0
  while( i < len(p1) and j < len(p2) ):
    if(p1[i] == p2[j]):
      answer.append(p1[i])
      i = i+1
      j = j+1
    elif(p1[i] < p2[j]):
      i = i+1
    else:
      j = j+1  
  return answer

result = intersect(index["brutus"], index["caesar"])
print (result)