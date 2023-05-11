#Given the head of a singly linked list, return true if it is a palindrome  or false otherwise.

#Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        ptr1 = head
        ptr2 = ptr1
        stack = []
        i = int(0)
        while(ptr1):
            stack.append(ptr1)
            ptr1 = ptr1.next
            i +=1
        i = i - 1
        while( i >= 0):
            if (stack[i].val == ptr2.val):
                i -= 1
                ptr2 = ptr2.next
            else:
             return False
        return True



