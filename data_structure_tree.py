import time
import random

from queue import LifoQueue, Queue
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        values = [str(self.val)]
        x = self
        while x.next:
            x = x.next
            values.append(str(x.val))
        return ", ".join(values)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.left.val if self.left else 'None'} <- {self.val} -> " \
               f"{self.right.val if self.right else 'None'} "

    def display(self):  # todo
        """prints a pretty version of the tree"""
        display_list = Solution.level_order_none_values(self)

        print(display_list)

    def generate_layers(self, prob, min_depth, max_depth):  # todo
        """
        overrides current tree with a new randomly generated, the lower "prob" the
         bigger will be the probability of the tree ending before reaching max
        :param prob: float [0, 1]
        :param min_depth: int
        :param max_depth: int
        :return: None
        """
        queue = Queue()
        queue.put((self, 0))
        cur_value = 0

        while not queue.empty():
            root, cur_level = queue.get()

            root.val = cur_value
            cur_value += 1
            if cur_level < max_depth:
                if cur_level > min_depth:
                    has_left = random.random() < prob
                    has_right = random.random() < prob
                else:
                    has_left, has_right = True, True

                if has_left:
                    new_root_left = TreeNode()
                    root.left = new_root_left
                    queue.put((new_root_left, cur_level + 1))
                if has_right:
                    new_root_right = TreeNode()
                    root.right = new_root_right
                    queue.put((new_root_right, cur_level + 1))


class Solution:
    @staticmethod
    def preorder_traversal_first_try(root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = LifoQueue()
        final_list = [root.val]
        is_left = True

        stack.put((True, root))  # go left = True
        while not stack.empty():
            if root.left is None and is_left:
                _, root = stack.get()
                is_left = False
                stack.put((False, root))

            elif root.right is None and not is_left:
                while not is_left:
                    if stack.empty():
                        break
                    is_left, root = stack.get()
                else:
                    is_left = False
                    stack.put((False, root))
            else:
                root = root.left if is_left else root.right
                is_left = True
                stack.put((True, root))  # go left = True
                final_list.append(root.val)

        return final_list

    @staticmethod
    def preorder_traversal(root: TreeNode) -> List[int]:
        """
        1. If there is no root, return an empty list
        2. Define a stack variable with the root inside it and a final_list that will be
         returned (both are lists)
        3. While there is something inside the stack:
        4. Append the value of the element to the final_list (there is no need to do
         anything with it)
        5. Add the right branch and after that the left branch, so the left branch will
        be traversed first.
        6. Return the final_list
        """
        if not root:  # 1
            return []

        stack, final_list = [root], []  # 2
        while stack:  # 3
            root = stack.pop()
            if root:
                final_list.append(root.val)  # 4
            if root.right:
                stack.append(root.right)  # 5
            if root.left:
                stack.append(root.left)
        return final_list  # 6

    @staticmethod
    def inorder_traversal(root: TreeNode) -> List[int]:
        """
        1. Return an empty list if there is no root, or it is None
        2. Define a stack and a final_list. The stack will have pairs of values ->
         (root: the data about the root; backtracking: True if the node has been already
         traversed)
        3. Get the root and the backtracking variable (again, it will be True if the
         root has been traversed to the left)
        4. Add the root value and continue if we are backtracking
        5. Add the right subtree (this will be the last to be accessed)
        6. Add the root node to the stack, so we can backtrack. Add the left subtree
         (this will be popped up in the next iteration of the loop)
        7. If we are on a left leaf, add the value (just to not have the work to add it
         to the stack again just to backtrack)
        8. Return the final_list
        """
        if not root:  # 1
            return []
        stack, final_list = [(root, False)], []  # 2
        while stack:
            root, backtracking = stack.pop()  # 3
            if backtracking:  # 4
                final_list.append(root.val)
                continue
            if root.right:
                stack.append((root.right, False))  # 5
            if root.left:
                stack.append((root, True))  # 6
                stack.append((root.left, False))
            else:
                final_list.append(root.val)  # 7
        return final_list  # 8

    @staticmethod
    def postorder_traversal(root: TreeNode) -> List[int]:
        """
        This is almost the same as the inorder traversal, but instead of adding the
         right subtree to the stack first, we add the root so we can backtrack, then add
         the right subtree and then the left, so we traverse first to the left, than
         to the right, and then backtrack."""
        if not root:
            return []
        stack, final_list = [(root, False)], []
        while stack:
            root, backtracking = stack.pop()
            if backtracking:
                final_list.append(root.val)
                continue
            stack.append((root, True))
            if root.right:
                stack.append((root.right, False))
            if root.left:
                stack.append((root.left, False))
        return final_list

    @staticmethod
    def level_order(root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = Queue()
        queue.put((root, 0))
        traversal, max_level = [[]], 0

        while not queue.empty():
            root, cur_level = queue.get()
            if cur_level > max_level:
                max_level = cur_level
                traversal.append([])

            traversal[cur_level].append(root.val)
            if root.left:
                queue.put((root.left, cur_level + 1))
            if root.right:
                queue.put((root.right, cur_level + 1))

        return traversal

    @staticmethod
    def level_order_none_values(root: TreeNode) -> List[List]:
        if not root:
            return []
        queue = Queue()
        queue.put((root, 0))
        traversal, root_max_depth = [[]], 0

        while not queue.empty():
            root, cur_level = queue.get()
            if cur_level > len(traversal) - 1:
                traversal.append([])

            if root:
                root_max_depth = cur_level
                traversal[cur_level].append(root.val)
                queue.put((root.left, cur_level + 1))
                queue.put((root.right, cur_level + 1))
            else:
                if cur_level - 1 > root_max_depth:
                    break
                traversal[cur_level].append(None)
                queue.put((None, cur_level + 1))
                queue.put((None, cur_level + 1))

        return traversal[:-2]

    def max_depth(self, root: TreeNode, depth=0) -> int:
        if root is None:
            return depth

        answer1 = self.max_depth(root.left, depth + 1)
        answer2 = self.max_depth(root.right, depth + 1)

        return max(answer1, answer2)

    @staticmethod
    def is_symmetric(root: TreeNode) -> bool:
        if root is None:
            return True
        stack = [(root.left, root.right)]
        answer = True
        while answer and stack:
            left_root, right_root = stack.pop()

            # if one of them is a root but the other isn't
            if bool(left_root) != bool(right_root):
                answer = False
            elif left_root is not None and right_root is not None:
                if left_root.val != right_root.val:
                    answer = False
                else:
                    stack += ((left_root.left, right_root.right),
                              (left_root.right, right_root.left))

        return answer

    @staticmethod
    def has_path_sum(root: TreeNode, target_sum) -> int:
        if root is None:
            return False
        stack = [(root, 0)]
        while stack:
            root, branch_value = stack.pop()
            cur_value = branch_value + root.val
            if root.right is None and root.left is None and cur_value == target_sum:
                return True
            if root.right:
                stack.append((root.right, cur_value))
            if root.left:
                stack.append((root.left, cur_value))
        return False

    @staticmethod
    def build_tree_iterative(inorder: List[int], postorder: List[int]) -> TreeNode:
        print(inorder, postorder)  # todo
        for value in inorder:
            pass


def main():
    nodes = [ListNode(val=value) for value in range(1, 5)]
    for i in range(0, len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    node_head = nodes[0]

    root = TreeNode()
    root.generate_layers(0.2, 1, 3)

    start_time = time.time()
    solution = Solution()
    root.display()
    print(solution.inorder_traversal(root))
    print("run_time:", time.time() - start_time)


if __name__ == "__main__":
    main()
