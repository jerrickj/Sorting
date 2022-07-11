from __future__ import annotations
import random
import re
import operator

# https://github.com/TheAlgorithms/Python/tree/master/sorts
# https://realpython.com/sorting-algorithms-python/#toc

def bead_sort(sequence: list) -> list:
    """
    Bead sort only works for sequences of non-negative integers.
    https://en.wikipedia.org/wiki/Bead_sort

    >>> bead_sort([6, 11, 12, 4, 1, 5])
    [1, 4, 5, 6, 11, 12]
    >>> bead_sort([9, 8, 7, 6, 5, 4 ,3, 2, 1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> bead_sort([5, 0, 4, 3])
    [0, 3, 4, 5]
    >>> bead_sort([8, 2, 1])
    [1, 2, 8]
    >>> bead_sort([1, .9, 0.0, 0, -1, -.9])
    Traceback (most recent call last):
    ...
    TypeError: Sequence must be list of non-negative integers
    >>> bead_sort("Hello world")
    Traceback (most recent call last):
    ...
    TypeError: Sequence must be list of non-negative integers
    """
    if any(not isinstance(x, int) or x < 0 for x in sequence):
        raise TypeError("Sequence must be list of non-negative integers")
    for _ in range(len(sequence)):
        for i, (rod_upper, rod_lower) in enumerate(zip(sequence, sequence[1:])):
            if rod_upper > rod_lower:
                sequence[i] -= rod_upper - rod_lower
                sequence[i + 1] += rod_upper - rod_lower
    return sequence

def bogo_sort(lst):
    """Pure implementation of the bogosort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: the same lst ordered by ascending
    Examples:
    >>> bogo_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> bogo_sort([])
    []
    >>> bogo_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    def is_sorted(lst):
        if len(lst) < 2:
            return True
        for i in range(len(lst) - 1):
            if lst[i] > lst[i + 1]:
                return False
        return True

    while not is_sorted(lst):
        random.shuffle(lst)
    return lst

def bubble_sort(array):
    n = len(array)

    for i in range(n):
        # Create a flag that will allow the function to
        # terminate early if there's nothing left to sort
        already_sorted = True

        # Start looking at each item of the list one by one,
        # comparing it with its adjacent value. With each
        # iteration, the portion of the array that you look at
        # shrinks because the remaining items have already been
        # sorted.
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                # If the item you're looking at is greater than its
                # adjacent value, then swap them
                array[j], array[j + 1] = array[j + 1], array[j]

                # Since you had to swap two elements,
                # set the `already_sorted` flag to `False` so the
                # algorithm doesn't finish prematurely
                already_sorted = False

        # If there were no swaps during the last iteration,
        # the array is already sorted, and you can terminate
        if already_sorted:
            break

    return array

def bitonic_sort(array: list[int], low: int, length: int, direction: int) -> None:
    """
    Python program for Bitonic Sort.
    Note that this program works only when size of input is a power of 2.
    """

    def comp_and_swap(array: list[int], index1: int, index2: int, direction: int) -> None:
        """Compare the value at given index1 and index2 of the array and swap them as per
        the given direction.
        The parameter direction indicates the sorting direction, ASCENDING(1) or
        DESCENDING(0); if (a[i] > a[j]) agrees with the direction, then a[i] and a[j] are
        interchanged.
        >>> arr = [12, 42, -21, 1]
        >>> comp_and_swap(arr, 1, 2, 1)
        >>> print(arr)
        [12, -21, 42, 1]
        >>> comp_and_swap(arr, 1, 2, 0)
        >>> print(arr)
        [12, 42, -21, 1]
        >>> comp_and_swap(arr, 0, 3, 1)
        >>> print(arr)
        [1, 42, -21, 12]
        >>> comp_and_swap(arr, 0, 3, 0)
        >>> print(arr)
        [12, 42, -21, 1]
        """
        if (direction == 1 and array[index1] > array[index2]) or (
            direction == 0 and array[index1] < array[index2]
        ):
            array[index1], array[index2] = array[index2], array[index1]

    def bitonic_merge(array: list[int], low: int, length: int, direction: int) -> None:
        """
        It recursively sorts a bitonic sequence in ascending order, if direction = 1, and in
        descending if direction = 0.
        The sequence to be sorted starts at index position low, the parameter length is the
        number of elements to be sorted.

        >>> arr = [12, 42, -21, 1]
        >>> bitonic_merge(arr, 0, 4, 1)
        >>> print(arr)
        [-21, 1, 12, 42]
        >>> bitonic_merge(arr, 0, 4, 0)
        >>> print(arr)
        [42, 12, 1, -21]
        """
        if length > 1:
            middle = int(length / 2)
            for i in range(low, low + middle):
                comp_and_swap(array, i, i + middle, direction)
            bitonic_merge(array, low, middle, direction)
            bitonic_merge(array, low + middle, middle, direction)

    """
    This function first produces a bitonic sequence by recursively sorting its two
    halves in opposite sorting orders, and then calls bitonic_merge to make them in the
    same order.
    >>> arr = [12, 34, 92, -23, 0, -121, -167, 145]
    >>> bitonic_sort(arr, 0, 8, 1)
    >>> arr
    [-167, -121, -23, 0, 12, 34, 92, 145]
    >>> bitonic_sort(arr, 0, 8, 0)
    >>> arr
    [145, 92, 34, 12, 0, -23, -121, -167]
    """
    if length > 1:
        middle = int(length / 2)
        bitonic_sort(array, low, middle, 1)
        bitonic_sort(array, low + middle, middle, 0)
        bitonic_merge(array, low, length, direction)

def bucket_sort(my_list: list) -> list:
    """
    >>> data = [-1, 2, -5, 0]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [9, 8, 7, 6, -12]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [.4, 1.2, .1, .2, -.9]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> bucket_sort([]) == sorted([])
    True
    >>> import random
    >>> lst = random.sample(range(-50, 50), 50)
    >>> bucket_sort(lst) == sorted(lst)
    True
    """
    if len(my_list) == 0:
        return []
    min_value, max_value = min(my_list), max(my_list)
    bucket_count = int(max_value - min_value) + 1
    buckets: list[list] = [[] for _ in range(bucket_count)]

    for i in my_list:
        buckets[int(i - min_value)].append(i)

    return [v for bucket in buckets for v in sorted(bucket)]

def cocktail_shaker_sort(unsorted: list) -> list:
    """
    Pure implementation of the cocktail shaker sort algorithm in Python.
    >>> cocktail_shaker_sort([4, 5, 2, 1, 2])
    [1, 2, 2, 4, 5]
    >>> cocktail_shaker_sort([-4, 5, 0, 1, 2, 11])
    [-4, 0, 1, 2, 5, 11]
    >>> cocktail_shaker_sort([0.1, -2.4, 4.4, 2.2])
    [-2.4, 0.1, 2.2, 4.4]
    >>> cocktail_shaker_sort([1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]
    >>> cocktail_shaker_sort([-4, -5, -24, -7, -11])
    [-24, -11, -7, -5, -4]
    """
    for i in range(len(unsorted) - 1, 0, -1):
        swapped = False

        for j in range(i, 0, -1):
            if unsorted[j] < unsorted[j - 1]:
                unsorted[j], unsorted[j - 1] = unsorted[j - 1], unsorted[j]
                swapped = True

        for j in range(i):
            if unsorted[j] > unsorted[j + 1]:
                unsorted[j], unsorted[j + 1] = unsorted[j + 1], unsorted[j]
                swapped = True

        if not swapped:
            break
    return unsorted

def comb_sort(data: list) -> list:
    """Pure implementation of comb sort algorithm in Python
    :param data: mutable lst with comparable items
    :return: the same lst in ascending order
    Examples:
    >>> comb_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> comb_sort([])
    []
    >>> comb_sort([99, 45, -7, 8, 2, 0, -15, 3])
    [-15, -7, 0, 2, 3, 8, 45, 99]
    """
    shrink_factor = 1.3
    gap = len(data)
    completed = False

    while not completed:

        # Update the gap value for a next comb
        gap = int(gap / shrink_factor)
        if gap <= 1:
            completed = True

        index = 0
        while index + gap < len(data):
            if data[index] > data[index + gap]:
                # Swap values
                data[index], data[index + gap] = data[index + gap], data[index]
                completed = False
            index += 1

    return data

def counting_sort(lst):
    """Pure implementation of counting sort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: the same lst ordered by ascending
    Examples:
    >>> counting_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> counting_sort([])
    []
    >>> counting_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    # if the lst is empty, returns empty
    if lst == []:
        return []

    # get some information about the lst
    coll_len = len(lst)
    coll_max = max(lst)
    coll_min = min(lst)

    # create the counting array
    counting_arr_length = coll_max + 1 - coll_min
    counting_arr = [0] * counting_arr_length

    # count how much a number appears in the lst
    for number in lst:
        counting_arr[number - coll_min] += 1

    # sum each position with it's predecessors. now, counting_arr[i] tells
    # us how many elements <= i has in the lst
    for i in range(1, counting_arr_length):
        counting_arr[i] = counting_arr[i] + counting_arr[i - 1]

    # create the output lst
    ordered = [0] * coll_len

    # place the elements in the output, respecting the original order (stable
    # sort) from end to begin, updating counting_arr
    for i in reversed(range(0, coll_len)):
        ordered[counting_arr[lst[i] - coll_min] - 1] = lst[i]
        counting_arr[lst[i] - coll_min] -= 1

    return ordered

def counting_sort_string(string):
    """
    >>> counting_sort_string("thisisthestring")
    'eghhiiinrsssttt'
    """
    return "".join([chr(i) for i in counting_sort([ord(c) for c in string])])

def cycle_sort(array: list) -> list:
    """
    >>> cycle_sort([4, 3, 2, 1])
    [1, 2, 3, 4]
    >>> cycle_sort([-4, 20, 0, -50, 100, -1])
    [-50, -4, -1, 0, 20, 100]
    >>> cycle_sort([-.1, -.2, 1.3, -.8])
    [-0.8, -0.2, -0.1, 1.3]
    >>> cycle_sort([])
    []
    """
    array_len = len(array)
    for cycle_start in range(0, array_len - 1):
        item = array[cycle_start]

        pos = cycle_start
        for i in range(cycle_start + 1, array_len):
            if array[i] < item:
                pos += 1

        if pos == cycle_start:
            continue

        while item == array[pos]:
            pos += 1

        array[pos], item = item, array[pos]
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, array_len):
                if array[i] < item:
                    pos += 1

            while item == array[pos]:
                pos += 1

            array[pos], item = item, array[pos]

    return array

def double_sort(lst):
    """This sorting algorithm sorts an array using the principle of bubble sort,
    but does it both from left to right and right to left.
    Hence, it's called "Double sort"
    :param lst: mutable ordered sequence of elements
    :return: the same lst in ascending order
    Examples:
    >>> double_sort([-1 ,-2 ,-3 ,-4 ,-5 ,-6 ,-7])
    [-7, -6, -5, -4, -3, -2, -1]
    >>> double_sort([])
    []
    >>> double_sort([-1 ,-2 ,-3 ,-4 ,-5 ,-6])
    [-6, -5, -4, -3, -2, -1]
    >>> double_sort([-3, 10, 16, -42, 29]) == sorted([-3, 10, 16, -42, 29])
    True
    """
    no_of_elements = len(lst)
    for i in range(
        0, int(((no_of_elements - 1) / 2) + 1)
    ):  # we don't need to traverse to end of list as
        for j in range(0, no_of_elements - 1):
            if (
                lst[j + 1] < lst[j]
            ):  # applying bubble sort algorithm from left to right (or forwards)
                temp = lst[j + 1]
                lst[j + 1] = lst[j]
                lst[j] = temp
            if (
                lst[no_of_elements - 1 - j] < lst[no_of_elements - 2 - j]
            ):  # applying bubble sort algorithm from right to left (or backwards)
                temp = lst[no_of_elements - 1 - j]
                lst[no_of_elements - 1 - j] = lst[no_of_elements - 2 - j]
                lst[no_of_elements - 2 - j] = temp
    return lst

def exchange_sort(numbers: list[int]) -> list[int]:
    """
    Uses exchange sort to sort a list of numbers.
    Source: https://en.wikipedia.org/wiki/Sorting_algorithm#Exchange_sort
    >>> exchange_sort([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    >>> exchange_sort([-1, -2, -3])
    [-3, -2, -1]
    >>> exchange_sort([1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]
    >>> exchange_sort([0, 10, -2, 5, 3])
    [-2, 0, 3, 5, 10]
    >>> exchange_sort([])
    []
    """
    numbers_length = len(numbers)
    for i in range(numbers_length):
        for j in range(i + 1, numbers_length):
            if numbers[j] < numbers[i]:
                numbers[i], numbers[j] = numbers[j], numbers[i]
    return numbers

def gnome_sort(lst: list) -> list:
    """
    Pure implementation of the gnome sort algorithm in Python
    Take some mutable ordered lst with heterogeneous comparable items inside as
    arguments, return the same lst ordered by ascending.
    Examples:
    >>> gnome_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> gnome_sort([])
    []
    >>> gnome_sort([-2, -5, -45])
    [-45, -5, -2]
    >>> "".join(gnome_sort(list(set("Gnomes are stupid!"))))
    ' !Gadeimnoprstu'
    """
    if len(lst) <= 1:
        return lst

    i = 1

    while i < len(lst):
        if lst[i - 1] <= lst[i]:
            i += 1
        else:
            lst[i - 1], lst[i] = lst[i], lst[i - 1]
            i -= 1
            if i == 0:
                i = 1

    return lst

def heap_sort(unsorted):
    """
    Pure implementation of the heap sort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: the same lst ordered by ascending
    Examples:
    >>> heap_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> heap_sort([])
    []
    >>> heap_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    def heapify(unsorted, index, heap_size):
        largest = index
        left_index = 2 * index + 1
        right_index = 2 * index + 2
        if left_index < heap_size and unsorted[left_index] > unsorted[largest]:
            largest = left_index

        if right_index < heap_size and unsorted[right_index] > unsorted[largest]:
            largest = right_index

        if largest != index:
            unsorted[largest], unsorted[index] = unsorted[index], unsorted[largest]
            heapify(unsorted, largest, heap_size)

    n = len(unsorted)
    for i in range(n // 2 - 1, -1, -1):
        heapify(unsorted, i, n)
    for i in range(n - 1, 0, -1):
        unsorted[0], unsorted[i] = unsorted[i], unsorted[0]
        heapify(unsorted, 0, i)
    return unsorted

def heapSort(arr):
    def heapify(arr, n, i):
        largest = i  # Initialize largest as root
        l = 2 * i + 1     # left = 2*i + 1
        r = 2 * i + 2     # right = 2*i + 2
    
        # See if left child of root exists and is
        # greater than root
        if l < n and arr[largest] < arr[l]:
            largest = l
    
        # See if right child of root exists and is
        # greater than root
        if r < n and arr[largest] < arr[r]:
            largest = r
    
        # Change root, if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap
    
            # Heapify the root.
            heapify(arr, n, largest)

    n = len(arr)
    # Build a maxheap.
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
  
    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)

    return arr

def insertion_sort(array, left=0, right=None):
    if right is None:
        right = len(array) - 1

    # Loop from the element indicated by
    # `left` until the element indicated by `right`
    for i in range(left + 1, right + 1):
        # This is the element we want to position in its
        # correct place
        key_item = array[i]

        # Initialize the variable that will be used to
        # find the correct position of the element referenced
        # by `key_item`
        j = i - 1

        # Run through the list of items (the left
        # portion of the array) and find the correct position
        # of the element referenced by `key_item`. Do this only
        # if the `key_item` is smaller than its adjacent values.
        while j >= left and array[j] > key_item:
            # Shift the value one position to the left
            # and reposition `j` to point to the next element
            # (from right to left)
            array[j + 1] = array[j]
            j -= 1

        # When you finish shifting the elements, position
        # the `key_item` in its correct location
        array[j + 1] = key_item

    return array

def iter_merge_sort(input_list: list) -> list:
    """
    Return a sorted copy of the input list
    >>> iter_merge_sort([5, 9, 8, 7, 1, 2, 7])
    [1, 2, 5, 7, 7, 8, 9]
    >>> iter_merge_sort([1])
    [1]
    >>> iter_merge_sort([2, 1])
    [1, 2]
    >>> iter_merge_sort([2, 1, 3])
    [1, 2, 3]
    >>> iter_merge_sort([4, 3, 2, 1])
    [1, 2, 3, 4]
    >>> iter_merge_sort([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    >>> iter_merge_sort(['c', 'b', 'a'])
    ['a', 'b', 'c']
    >>> iter_merge_sort([0.3, 0.2, 0.1])
    [0.1, 0.2, 0.3]
    >>> iter_merge_sort(['dep', 'dang', 'trai'])
    ['dang', 'dep', 'trai']
    >>> iter_merge_sort([6])
    [6]
    >>> iter_merge_sort([])
    []
    >>> iter_merge_sort([-2, -9, -1, -4])
    [-9, -4, -2, -1]
    >>> iter_merge_sort([1.1, 1, 0.0, -1, -1.1])
    [-1.1, -1, 0.0, 1, 1.1]
    >>> iter_merge_sort(['c', 'b', 'a'])
    ['a', 'b', 'c']
    >>> iter_merge_sort('cba')
    ['a', 'b', 'c']
    """
    def merge(input_list: list, low: int, mid: int, high: int) -> list:
        """
        sorting left-half and right-half individually
        then merging them into result
        """
        result = []
        left, right = input_list[low:mid], input_list[mid : high + 1]
        while left and right:
            result.append((left if left[0] <= right[0] else right).pop(0))
        input_list[low : high + 1] = result + left + right
        return input_list

    if len(input_list) <= 1:
        return input_list
    input_list = list(input_list)

    # iteration for two-way merging
    p = 2
    while p <= len(input_list):
        # getting low, high and middle value for merge-sort of single list
        for i in range(0, len(input_list), p):
            low = i
            high = i + p - 1
            mid = (low + high + 1) // 2
            input_list = merge(input_list, low, mid, high)
        # final merge of last two parts
        if p * 2 >= len(input_list):
            mid = i
            input_list = merge(input_list, 0, mid, len(input_list) - 1)
            break
        p *= 2

    return input_list

def merge(left, right):
    # Function to merge two different arrays - not sort them

    # If the first array is empty, then nothing needs
    # to be merged, and you can return the second array as the result
    if len(left) == 0:
        return right
    # If the second array is empty, then nothing needs
    # to be merged, and you can return the first array as the result
    if len(right) == 0:
        return left
    result = []
    index_left = index_right = 0
    # Now go through both arrays until all the elements
    # make it into the resultant array
    while len(result) < len(left) + len(right):
        # The elements need to be sorted to add them to the
        # resultant array, so you need to decide whether to get
        # the next element from the first or the second array
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
        # If you reach the end of either array, then you can
        # add the remaining elements from the other array to
        # the result and break the loop
        if index_right == len(right):
            result += left[index_left:]
            break
        if index_left == len(left):
            result += right[index_right:]
            break
    return result

def merge_sort(lst: list) -> list:
    def merge(left: list, right: list) -> list:
        """merge left and right
        :param left: left lst
        :param right: right lst
        :return: merge result
        """
        def _merge():
            while left and right:
                yield (left if left[0] <= right[0] else right).pop(0)
            yield from left
            yield from right

        return list(_merge())

    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    return merge(merge_sort(lst[:mid]), merge_sort(lst[mid:]))

def unknown_merge_sort(lst):
    """Pure implementation of the fastest merge sort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: a lst ordered by ascending
    Examples:
    >>> merge_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> merge_sort([])
    []
    >>> merge_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    start, end = [], []
    while len(lst) > 1:
        min_one, max_one = min(lst), max(lst)
        start.append(min_one)
        end.append(max_one)
        lst.remove(min_one)
        lst.remove(max_one)
    end.reverse()
    return start + lst + end

def merge_insertion_sort(lst: list[int]) -> list[int]:
    """Pure implementation of merge-insertion sort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: the same lst ordered by ascending
    Examples:
    >>> merge_insertion_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> merge_insertion_sort([99])
    [99]
    >>> merge_insertion_sort([-2, -5, -45])
    [-45, -5, -2]
    Testing with all permutations on range(0,5):
    >>> import itertools
    >>> permutations = list(itertools.permutations([0, 1, 2, 3, 4]))
    >>> all(merge_insertion_sort(p) == [0, 1, 2, 3, 4] for p in permutations)
    True
    """

    def binary_search_insertion(sorted_list, item):
        left = 0
        right = len(sorted_list) - 1
        while left <= right:
            middle = (left + right) // 2
            if left == right:
                if sorted_list[middle] < item:
                    left = middle + 1
                break
            elif sorted_list[middle] < item:
                left = middle + 1
            else:
                right = middle - 1
        sorted_list.insert(left, item)
        return sorted_list

    def sortlist_2d(list_2d):
        def merge(left, right):
            result = []
            while left and right:
                if left[0][0] < right[0][0]:
                    result.append(left.pop(0))
                else:
                    result.append(right.pop(0))
            return result + left + right

        length = len(list_2d)
        if length <= 1:
            return list_2d
        middle = length // 2
        return merge(sortlist_2d(list_2d[:middle]), sortlist_2d(list_2d[middle:]))

    if len(lst) <= 1:
        return lst

    """
    Group the items into two pairs, and leave one element if there is a last odd item.
    Example: [999, 100, 75, 40, 10000]
                -> [999, 100], [75, 40]. Leave 10000.
    """
    two_paired_list = []
    has_last_odd_item = False
    for i in range(0, len(lst), 2):
        if i == len(lst) - 1:
            has_last_odd_item = True
        else:
            """
            Sort two-pairs in each groups.
            Example: [999, 100], [75, 40]
                        -> [100, 999], [40, 75]
            """
            if lst[i] < lst[i + 1]:
                two_paired_list.append([lst[i], lst[i + 1]])
            else:
                two_paired_list.append([lst[i + 1], lst[i]])

    """
    Sort two_paired_list.
    Example: [100, 999], [40, 75]
                -> [40, 75], [100, 999]
    """
    sorted_list_2d = sortlist_2d(two_paired_list)

    """
    40 < 100 is sure because it has already been sorted.
    Generate the sorted_list of them so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           40     100
           75     999
        ->
           group0 group1
           [40,   100]
           75     999
    """
    result = [i[0] for i in sorted_list_2d]

    """
    100 < 999 is sure because it has already been sorted.
    Put 999 in last of the sorted_list so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           [40,   100]
           75     999
        ->
           group0 group1
           [40,   100,   999]
           75
    """
    result.append(sorted_list_2d[-1][1])

    """
    Insert the last odd item left if there is.
    Example:
           group0 group1
           [40,   100,   999]
           75
        ->
           group0 group1
           [40,   100,   999,   10000]
           75
    """
    if has_last_odd_item:
        pivot = lst[-1]
        result = binary_search_insertion(result, pivot)

    """
    Insert the remaining items.
    In this case, 40 < 75 is sure because it has already been sorted.
    Therefore, you only need to insert 75 into [100, 999, 10000],
    so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           [40,   100,   999,   10000]
            ^ You don't need to compare with this as 40 < 75 is already sure.
           75
        ->
           [40,   75,    100,   999,   10000]
    """
    is_last_odd_item_inserted_before_this_index = False
    for i in range(len(sorted_list_2d) - 1):
        if result[i] == lst[-1] and has_last_odd_item:
            is_last_odd_item_inserted_before_this_index = True
        pivot = sorted_list_2d[i][1]
        # If last_odd_item is inserted before the item's index,
        # you should forward index one more.
        if is_last_odd_item_inserted_before_this_index:
            result = result[: i + 2] + binary_search_insertion(result[i + 2 :], pivot)
        else:
            result = result[: i + 1] + binary_search_insertion(result[i + 1 :], pivot)

    return result

def natural_sort(input_list: list[str]) -> list[str]:
    """
    Sort the given list of strings in the way that humans expect.
    The normal Python sort algorithm sorts lexicographically,
    so you might not get the results that you expect...
    >>> example1 = ['2 ft 7 in', '1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '7 ft 6 in']
    >>> sorted(example1)
    ['1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '2 ft 7 in', '7 ft 6 in']
    >>> # The natural sort algorithm sort based on meaning and not computer code point.
    >>> natural_sort(example1)
    ['1 ft 5 in', '2 ft 7 in', '2 ft 11 in', '7 ft 6 in', '10 ft 2 in']
    >>> example2 = ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> sorted(example2)
    ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> natural_sort(example2)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']
    """

    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in re.split("([0-9]+)", key)]

    return sorted(input_list, key=alphanum_key)

def odd_even_sort(input_list: list) -> list:
    """this algorithm uses the same idea of bubblesort,
    but by first dividing in two phase (odd and even).
    Originally developed for use on parallel processors
    with local interconnections.
    :param lst: mutable ordered sequence of elements
    :return: same lst in ascending order
    Examples:
    >>> odd_even_sort([5 , 4 ,3 ,2 ,1])
    [1, 2, 3, 4, 5]
    >>> odd_even_sort([])
    []
    >>> odd_even_sort([-10 ,-1 ,10 ,2])
    [-10, -1, 2, 10]
    >>> odd_even_sort([1 ,2 ,3 ,4])
    [1, 2, 3, 4]
    """
    sorted = False
    while sorted is False:  # Until all the indices are traversed keep looping
        sorted = True
        for i in range(0, len(input_list) - 1, 2):  # iterating over all even indices
            if input_list[i] > input_list[i + 1]:

                input_list[i], input_list[i + 1] = input_list[i + 1], input_list[i]
                # swapping if elements not in order
                sorted = False

        for i in range(1, len(input_list) - 1, 2):  # iterating over all odd indices
            if input_list[i] > input_list[i + 1]:
                input_list[i], input_list[i + 1] = input_list[i + 1], input_list[i]
                # swapping if elements not in order
                sorted = False
    return input_list

def odd_even_transposition(arr: list) -> list:
    """
    >>> odd_even_transposition([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    >>> odd_even_transposition([13, 11, 18, 0, -1])
    [-1, 0, 11, 13, 18]
    >>> odd_even_transposition([-.1, 1.1, .1, -2.9])
    [-2.9, -0.1, 0.1, 1.1]
    """
    arr_size = len(arr)
    for _ in range(arr_size):
        for i in range(_ % 2, arr_size - 1, 2):
            if arr[i + 1] < arr[i]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]

    return arr

def pancake_sort(arr):
    """Sort Array with Pancake Sort.
    :param arr: lst containing comparable items
    :return: lst ordered in ascending order of items
    Examples:
    >>> pancake_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> pancake_sort([])
    []
    >>> pancake_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    cur = len(arr)
    while cur > 1:
        # Find the maximum number in arr
        mi = arr.index(max(arr[0:cur]))
        # Reverse from 0 to mi
        arr = arr[mi::-1] + arr[mi + 1 : len(arr)]
        # Reverse whole list
        arr = arr[cur - 1 :: -1] + arr[cur : len(arr)]
        cur -= 1
    return arr

def pigeon_sort(array: list[int]) -> list[int]:
    """
    Implementation of pigeon hole sort algorithm
    :param array: lst of comparable items
    :return: lst sorted in ascending order
    >>> pigeon_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> pigeon_sort([])
    []
    >>> pigeon_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    if len(array) == 0:
        return array

    _min, _max = min(array), max(array)

    # Compute the variables
    holes_range = _max - _min + 1
    holes, holes_repeat = [0] * holes_range, [0] * holes_range

    # Make the sorting.
    for i in array:
        index = i - _min
        holes[index] = i
        holes_repeat[index] += 1

    # Makes the array back by replacing the numbers.
    index = 0
    for i in range(holes_range):
        while holes_repeat[i] > 0:
            array[index] = holes[i]
            index += 1
            holes_repeat[i] -= 1

    # Returns the sorted array.
    return array

def pigeonhole_sort(a):
    """
    >>> a = [8, 3, 2, 7, 4, 6, 8]
    >>> b = sorted(a)  # a nondestructive sort
    >>> pigeonhole_sort(a)  # a destructive sort
    >>> a == b
    True
    """
    # size of range of values in the list (ie, number of pigeonholes we need)

    min_val = min(a)  # min() finds the minimum value
    max_val = max(a)  # max() finds the maximum value

    size = max_val - min_val + 1  # size is difference of max and min values plus one

    # list of pigeonholes of size equal to the variable size
    holes = [0] * size

    # Populate the pigeonholes.
    for x in a:
        assert isinstance(x, int), "integers only please"
        holes[x - min_val] += 1

    # Putting the elements back into the array in an order.
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            a[i] = count + min_val
            i += 1

def selection_sort(lst):
    """Pure implementation of the selection sort algorithm in Python
    :param lst: some mutable ordered lst with heterogeneous
    comparable items inside
    :return: the same lst ordered by ascending
    Examples:
    >>> selection_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> selection_sort([])
    []
    >>> selection_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    length = len(lst)
    for i in range(length - 1):
        least = i
        for k in range(i + 1, length):
            if lst[k] < lst[least]:
                least = k
        if least != i:
            lst[least], lst[i] = (lst[i], lst[least])
    return lst

def timsort(array):
    min_run = 32
    n = len(array)

    # Start by slicing and sorting small portions of the
    # input array. The size of these slices is defined by
    # your `min_run` size.
    for i in range(0, n, min_run):
        insertion_sort(array, i, min((i + min_run - 1), n - 1))

    # Now you can start merging the sorted slices.
    # Start from `min_run`, doubling the size on
    # each iteration until you surpass the length of
    # the array.
    size = min_run
    while size < n:
        # Determine the arrays that will
        # be merged together
        for start in range(0, n, size * 2):
            # Compute the `midpoint` (where the first array ends
            # and the second starts) and the `endpoint` (where
            # the second array ends)
            midpoint = start + size - 1
            end = min((start + size * 2 - 1), (n-1))

            # Merge the two subarrays.
            # The `left` array should go from `start` to
            # `midpoint + 1`, while the `right` array should
            # go from `midpoint + 1` to `end + 1`.
            merged_array = merge(
                left=array[start:midpoint + 1],
                right=array[midpoint + 1:end + 1])

            # Finally, put the merged array back into
            # your array
            array[start:start + len(merged_array)] = merged_array

        # Each iteration should double the size of your arrays
        size *= 2

    return array

def tim_sort(array): 
    MINIMUM= 32 # user defined 'minrun' variable; default = 32
    def find_minrun(n): 
  
        r = 0
        while n >= MINIMUM: 
            r |= n & 1
            n >>= 1
        return n + r 
  
    def insertion_sort(array, left, right): 
        for i in range(left+1,right+1):
            element = array[i]
            j = i-1
            while element<array[j] and j>=left :
                array[j+1] = array[j]
                j -= 1
            array[j+1] = element
        return array

    def merge_sort(array, l, m, r): 
    
        array_length1= m - l + 1
        array_length2 = r - m 
        left = []
        right = []
        for i in range(0, array_length1): 
            left.append(array[l + i]) 
        for i in range(0, array_length2): 
            right.append(array[m + 1 + i]) 
    
        i=0
        j=0
        k=l
    
        while j < array_length2 and  i < array_length1: 
            if left[i] <= right[j]: 
                array[k] = left[i] 
                i += 1
    
            else: 
                array[k] = right[j] 
                j += 1
    
            k += 1
    
        while i < array_length1: 
            array[k] = left[i] 
            k += 1
            i += 1
    
        while j < array_length2: 
            array[k] = right[j] 
            k += 1
            j += 1
    
    n = len(array) 
    minrun = find_minrun(n) 
  
    for start in range(0, n, minrun): 
        end = min(start + minrun - 1, n - 1) 
        insertion_sort(array, start, end) 
   
    size = minrun 
    while size < n: 
  
        for left in range(0, n, 2 * size): 
  
            mid = min(n - 1, left + size - 1) 
            right = min((left + 2 * size - 1), (n - 1)) 
            merge_sort(array, left, mid, right) 
  
        size = 2 * size

def quick_sort(lst: list) -> list:
    """A pure Python implementation of quick sort algorithm
    :param lst: a mutable lst of comparable items
    :return: the same lst ordered by ascending
    Examples:
    >>> quick_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> quick_sort([])
    []
    >>> quick_sort([-2, 5, 0, -45])
    [-45, -2, 0, 5]
    """
    if len(lst) < 2:
        return lst
    pivot = lst.pop()  # Use the last element as the first pivot
    greater: list[int] = []  # All elements greater than pivot
    lesser: list[int] = []  # All elements less than or equal to pivot
    for element in lst:
        (greater if element > pivot else lesser).append(element)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)

def three_way_radix_quicksort(sorting: list) -> list:
    """
    Three-way radix quicksort:
    https://en.wikipedia.org/wiki/Quicksort#Three-way_radix_quicksort
    First divide the list into three parts.
    Then recursively sort the "less than" and "greater than" partitions.
    >>> three_way_radix_quicksort([])
    []
    >>> three_way_radix_quicksort([1])
    [1]
    >>> three_way_radix_quicksort([-5, -2, 1, -2, 0, 1])
    [-5, -2, -2, 0, 1, 1]
    >>> three_way_radix_quicksort([1, 2, 5, 1, 2, 0, 0, 5, 2, -1])
    [-1, 0, 0, 1, 1, 2, 2, 2, 5, 5]
    """
    if len(sorting) <= 1:
        return sorting
    return (
        three_way_radix_quicksort([i for i in sorting if i < sorting[0]])
        + [i for i in sorting if i == sorting[0]]
        + three_way_radix_quicksort([i for i in sorting if i > sorting[0]]))

def quick_sort_lomuto_partition(sorting: list, left: int, right: int) -> None:
    """
    A pure Python implementation of quick sort algorithm(in-place)
    with Lomuto partition scheme:
    https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme
    :param sorting: sort list
    :param left: left endpoint of sorting
    :param right: right endpoint of sorting
    :return: None
    Examples:
    >>> nums1 = [0, 5, 3, 1, 2]
    >>> quick_sort_lomuto_partition(nums1, 0, 4)
    >>> nums1
    [0, 1, 2, 3, 5]
    >>> nums2 = []
    >>> quick_sort_lomuto_partition(nums2, 0, 0)
    >>> nums2
    []
    >>> nums3 = [-2, 5, 0, -4]
    >>> quick_sort_lomuto_partition(nums3, 0, 3)
    >>> nums3
    [-4, -2, 0, 5]
    """
    def lomuto_partition(sorting: list, left: int, right: int) -> int:
        """
        Example:
        >>> lomuto_partition([1,5,7,6], 0, 3)
        2
        """
        pivot = sorting[right]
        store_index = left
        for i in range(left, right):
            if sorting[i] < pivot:
                sorting[store_index], sorting[i] = sorting[i], sorting[store_index]
                store_index += 1
        sorting[right], sorting[store_index] = sorting[store_index], sorting[right]
        return store_index

    if left < right:
        pivot_index = lomuto_partition(sorting, left, right)
        quick_sort_lomuto_partition(sorting, left, pivot_index - 1)
        quick_sort_lomuto_partition(sorting, pivot_index + 1, right)

def radix_sort(list_of_ints: list[int]) -> list[int]:
    """
    Examples:
    >>> radix_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> radix_sort(list(range(15))) == sorted(range(15))
    True
    >>> radix_sort(list(range(14,-1,-1))) == sorted(range(15))
    True
    >>> radix_sort([1,100,10,1000]) == sorted([1,100,10,1000])
    True
    """
    RADIX = 10
    placement = 1
    max_digit = max(list_of_ints)
    while placement <= max_digit:
        # declare and initialize empty buckets
        buckets: list[list] = [list() for _ in range(RADIX)]
        # split list_of_ints between the buckets
        for i in list_of_ints:
            tmp = int((i / placement) % RADIX)
            buckets[tmp].append(i)
        # put each buckets' contents into list_of_ints
        a = 0
        for b in range(RADIX):
            for i in buckets[b]:
                list_of_ints[a] = i
                a += 1
        # move to next
        placement *= RADIX
    return list_of_ints

def recursive_bubble_sort(list_data: list, length: int = 0) -> list:
    """
    It is similar is bubble sort but recursive.
    :param list_data: mutable ordered sequence of elements
    :param length: length of list data
    :return: the same list in ascending order
    >>> bubble_sort([0, 5, 2, 3, 2], 5)
    [0, 2, 2, 3, 5]
    >>> bubble_sort([], 0)
    []
    >>> bubble_sort([-2, -45, -5], 3)
    [-45, -5, -2]
    >>> bubble_sort([-23, 0, 6, -4, 34], 5)
    [-23, -4, 0, 6, 34]
    >>> bubble_sort([-23, 0, 6, -4, 34], 5) == sorted([-23, 0, 6, -4, 34])
    True
    >>> bubble_sort(['z','a','y','b','x','c'], 6)
    ['a', 'b', 'c', 'x', 'y', 'z']
    >>> bubble_sort([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6])
    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
    """
    length = length or len(list_data)
    swapped = False
    for i in range(length - 1):
        if list_data[i] > list_data[i + 1]:
            list_data[i], list_data[i + 1] = list_data[i + 1], list_data[i]
            swapped = True

    return list_data if not swapped else bubble_sort(list_data, length - 1)

def rec_insertion_sort(lst: list, n: int):
    """
    Given a lst of numbers and its length, sorts the lsts
    in ascending order
    :param lst: A mutable lst of comparable elements
    :param n: The length of lsts
    >>> col = [1, 2, 1]
    >>> rec_insertion_sort(col, len(col))
    >>> print(col)
    [1, 1, 2]
    >>> col = [2, 1, 0, -1, -2]
    >>> rec_insertion_sort(col, len(col))
    >>> print(col)
    [-2, -1, 0, 1, 2]
    >>> col = [1]
    >>> rec_insertion_sort(col, len(col))
    >>> print(col)
    [1]
    """
    def insert_next(lst: list, index: int):
        """
        Inserts the '(index-1)th' element into place
        >>> col = [3, 2, 4, 2]
        >>> insert_next(col, 1)
        >>> print(col)
        [2, 3, 4, 2]
        >>> col = [3, 2, 3]
        >>> insert_next(col, 2)
        >>> print(col)
        [3, 2, 3]
        >>> col = []
        >>> insert_next(col, 1)
        >>> print(col)
        []
        """
        # Checks order between adjacent elements
        if index >= len(lst) or lst[index - 1] <= lst[index]:
            return

        # Swaps adjacent elements since they are not in ascending order
        lst[index - 1], lst[index] = (
            lst[index],
            lst[index - 1],
        )

        insert_next(lst, index + 1)

    # Checks if the entire lst has been sorted
    if len(lst) <= 1 or n <= 1:
        return

    insert_next(lst, n - 1)
    rec_insertion_sort(lst, n - 1)

def recursive_merge(arr: list[int]) -> list[int]:
    """Return a sorted array.
    >>> recursive_merge([10,9,8,7,6,5,4,3,2,1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> recursive_merge([1,2,3,4,5,6,7,8,9,10])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> recursive_merge([10,22,1,2,3,9,15,23])
    [1, 2, 3, 9, 10, 15, 22, 23]
    >>> recursive_merge([100])
    [100]
    >>> recursive_merge([])
    []
    """
    if len(arr) > 1:
        middle_length = len(arr) // 2  # Finds the middle of the array
        left_array = arr[
            :middle_length
        ]  # Creates an array of the elements in the first half.
        right_array = arr[
            middle_length:
        ]  # Creates an array of the elements in the second half.
        left_size = len(left_array)
        right_size = len(right_array)
        recursive_merge(left_array)  # Starts sorting the left.
        recursive_merge(right_array)  # Starts sorting the right
        left_index = 0  # Left Counter
        right_index = 0  # Right Counter
        index = 0  # Position Counter
        while (
            left_index < left_size and right_index < right_size
        ):  # Runs until the lowers size of the left and right are sorted.
            if left_array[left_index] < right_array[right_index]:
                arr[index] = left_array[left_index]
                left_index += 1
            else:
                arr[index] = right_array[right_index]
                right_index += 1
            index += 1
        while (
            left_index < left_size
        ):  # Adds the left over elements in the left half of the array
            arr[index] = left_array[left_index]
            left_index += 1
            index += 1
        while (
            right_index < right_size
        ):  # Adds the left over elements in the right half of the array
            arr[index] = right_array[right_index]
            right_index += 1
            index += 1
    return arr

def shell_sort(lst):
    """Pure implementation of shell sort algorithm in Python
    :param lst:  Some mutable ordered lst with heterogeneous
    comparable items inside
    :return:  the same lst ordered by ascending
    >>> shell_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> shell_sort([])
    []
    >>> shell_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    # Marcin Ciura's gap sequence

    gaps = [701, 301, 132, 57, 23, 10, 4, 1]
    for gap in gaps:
        for i in range(gap, len(lst)):
            insert_value = lst[i]
            j = i
            while j >= gap and lst[j - gap] > insert_value:
                lst[j] = lst[j - gap]
                j -= gap
            if j != i:
                lst[j] = insert_value
    return lst

def slowsort(sequence: list, start: int | None = None, end: int | None = None) -> None:
    """
    Sorts sequence[start..end] (both inclusive) in-place.
    start defaults to 0 if not given.
    end defaults to len(sequence) - 1 if not given.
    It returns None.
    >>> seq = [1, 6, 2, 5, 3, 4, 4, 5]; slowsort(seq); seq
    [1, 2, 3, 4, 4, 5, 5, 6]
    >>> seq = []; slowsort(seq); seq
    []
    >>> seq = [2]; slowsort(seq); seq
    [2]
    >>> seq = [1, 2, 3, 4]; slowsort(seq); seq
    [1, 2, 3, 4]
    >>> seq = [4, 3, 2, 1]; slowsort(seq); seq
    [1, 2, 3, 4]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, 2, 7); seq
    [9, 8, 2, 3, 4, 5, 6, 7, 1, 0]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, end = 4); seq
    [5, 6, 7, 8, 9, 4, 3, 2, 1, 0]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, start = 5); seq
    [9, 8, 7, 6, 5, 0, 1, 2, 3, 4]
    """
    if start is None:
        start = 0

    if end is None:
        end = len(sequence) - 1

    if start >= end:
        return

    mid = (start + end) // 2

    slowsort(sequence, start, mid)
    slowsort(sequence, mid + 1, end)

    if sequence[end] < sequence[mid]:
        sequence[end], sequence[mid] = sequence[mid], sequence[end]

    slowsort(sequence, start, end - 1)

def stooge_sort(arr):
    """
    Examples:
    >>> stooge_sort([18.1, 0, -7.1, -1, 2, 2])
    [-7.1, -1, 0, 2, 2, 18.1]
    >>> stooge_sort([])
    []
    """

    def stooge(arr, i, h):
        if i >= h:
            return

        # If first element is smaller than the last then swap them
        if arr[i] > arr[h]:
            arr[i], arr[h] = arr[h], arr[i]

        # If there are more than 2 elements in the array
        if h - i + 1 > 2:
            t = (int)((h - i + 1) / 3)

            # Recursively sort first 2/3 elements
            stooge(arr, i, (h - t))

            # Recursively sort last 2/3 elements
            stooge(arr, i + t, (h))

            # Recursively sort first 2/3 elements
            stooge(arr, i, (h - t))

    stooge(arr, 0, len(arr) - 1)
    return arr

def strand_sort(arr: list, reverse: bool = False, solution: list = None) -> list:
    """
    Strand sort implementation
    source: https://en.wikipedia.org/wiki/Strand_sort
    :param arr: Unordered input list
    :param reverse: Descent ordering flag
    :param solution: Ordered items container
    Examples:
    >>> strand_sort([4, 2, 5, 3, 0, 1])
    [0, 1, 2, 3, 4, 5]
    >>> strand_sort([4, 2, 5, 3, 0, 1], reverse=True)
    [5, 4, 3, 2, 1, 0]
    """
    _operator = operator.lt if reverse else operator.gt
    solution = solution or []

    if not arr:
        return solution

    sublist = [arr.pop(0)]
    for i, item in enumerate(arr):
        if _operator(item, sublist[-1]):
            sublist.append(item)
            arr.pop(i)

    #  merging sublist into solution list
    if not solution:
        solution.extend(sublist)
    else:
        while sublist:
            item = sublist.pop(0)
            for i, xx in enumerate(solution):
                if not _operator(item, xx):
                    solution.insert(i, item)
                    break
            else:
                solution.append(item)

    strand_sort(arr, reverse, solution)
    return solution

def topological_sort(start, visited, sort):
    edges = {"a": ["c", "b"], "b": ["d", "e"], "c": [], "d": [], "e": []}
    vertices = ["a", "b", "c", "d", "e"]

    """Perform topological sort on a directed acyclic graph."""
    current = start
    # add current to visited
    visited.append(current)
    neighbors = edges[current]
    for neighbor in neighbors:
        # if neighbor not in visited, visit
        if neighbor not in visited:
            sort = topological_sort(neighbor, visited, sort)
    # if all neighbors visited add current to sort
    sort.append(current)
    # if all vertices haven't been visited select a new one to visit
    if len(visited) != len(vertices):
        for vertice in vertices:
            if vertice not in visited:
                sort = topological_sort(vertice, visited, sort)
    # return sort
    return sort

def wiggle_sort(nums: list) -> list:
    """
    Python implementation of wiggle.
    Example:
    >>> wiggle_sort([0, 5, 3, 2, 2])
    [0, 5, 2, 3, 2]
    >>> wiggle_sort([])
    []
    >>> wiggle_sort([-2, -5, -45])
    [-45, -2, -5]
    >>> wiggle_sort([-2.1, -5.68, -45.11])
    [-45.11, -2.1, -5.68]
    """
    for i, _ in enumerate(nums):
        if (i % 2 == 1) == (nums[i - 1] > nums[i]):
            nums[i - 1], nums[i] = nums[i], nums[i - 1]

    return nums

