
# Remove Nth Node From End of List

class ListNode:
    def __init__(self, val=0, next=None):
       self.val = val
        self.next = next


def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head
    fast_pointer = dummy
    slow_pointer = dummy
    count = 1
    while count < n:
        count += 1
        fast_pointer = fast_pointer.next
    prev = slow_pointer
    while fast_pointer.next:
        fast_pointer = fast_pointer.next
        prev = slow_pointer
        slow_pointer = slow_pointer.next

    prev.next = slow_pointer.next
    return dummy.next


"""
if head == None:
    return head
if (n < 0 or n > 100):
    return head

node = head
prev = head
if (node == None):
    return head
l = 0
nodeDict = {}
while (node != None):
    curr = node
    # print("l:", l,"for Node val :", curr.val)
    node = node.next

    nodeDict[l] = curr
    l = l + 1

delNode = l - n
print("lenghth, n , l-1", l, n, delNode)

if (delNode == 0 or l == n):
    head = head.next


elif delNode > 0:
    prev = nodeDict[delNode - 1]
    if prev.next != None:
        prev.next = (prev.next).next
    else:
        prev.next = None

return head
"""
