from tqdm import tqdm

class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

        # stores the edges to the arg1 key
        # a list of trippels ['arg1', 'arg2', 'rel']
        self.edges = []

        self.embedding = []


class Trie(object):
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")

    def insert_dataset(self, d_set, embs):
        for index, data in tqdm(enumerate(d_set), total=len(d_set)):
            self.insert(data['arg1'], [data['arg1'], data['arg2'], data['rel']], embs[index])

    def insert(self, word, edge, embedding):
        """Insert a word into the trie"""
        node = self.root
        split_word = word.split('/')
        actual_word = split_word[3]

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in actual_word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node

        # Mark the end of a word
        node.is_end = True

        # Insert edge to edges list
        node.edges.append(edge)

        node.embedding.append(embedding)

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return [], []

        return node.edges, node.embedding
