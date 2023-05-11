#Write a function to find the longest common prefix string amongst an array of strings.
#If there is no common prefix, return an empty string "".

strs = ["aaa","aa","aaa"]
#strs = ["dog","racecar","car"]
#strs = ["flower","flow","flight"]
#strs = ["aac","acab","aa","abba","aa"]
strs = ["flower","flawer","flvwer","flower"]
notFound = 0

l = len(strs[0])
length = l
print ("length", l)
for i in range(1, len(strs)):
        print(" String :", i, ": :", strs[i])
        length = min(l, len(strs[i]))
        print ("length", length)
        if (strs[0][0:length] == strs[i][0:length]):
            l = length
        else :
            while (length > 0 and strs[0][0:length] !=strs[i][0:length]):
                length = length - 1
                print ("length :", length, "Strings :", strs[0][0:length], ": ", strs[i][0:length])
        print("length", length)
        if length == 0:
            notFound = 1
            break
        else :
            l = length


if notFound == 0:
    print(strs[0][0:length])
else :
    print ("Not Found")