class TrieNode:
    def __init__(self):
        self.is_end_of_word = False
        self.children = {}
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.has_words = False
    
    def insert(self, word: str):

        self.has_words = True
        curr = self.root

        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_end_of_word = True

    def search(self, word: str) -> bool:
        curr = self.root

        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        
        return curr.is_end_of_word

    def starts_with(self, prefix: str) -> bool:

        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        
        return True
            
def trie_operations(words, queries):
    # words: list of strings to insert
    # queries: list of (op, value) tuples where op is 'search' or 'startsWith'
    # return: list of booleans, one per query
    
    trie = Trie()

    for word in words:
        trie.insert(word)
        
    results = []

    for op, value in queries:
        if op == "search":
            results.append(trie.search(value))
        else:
            results.append(trie.starts_with(value))
    
    return results