#By Steve Hanov, 2011. Released to the public domain
import time
import sys
from trie_conceptnet import Trie
from Graph_builder import GraphBuilder





# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search( word , maxCost, trie ):

    # build first row
    currentRow = range( len(word) + 1 )

    results = []
    root = trie.root
    # recursively search each branch of the trie
    for letter in root.children:
        searchRecursive( root.children[letter], letter, word, currentRow,
            results, maxCost )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already.
def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in range( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word != None:
        results.append( (node.word, currentRow[-1] ) )

    # if any entries in the row are less than the maximum cost, then
    # recursively search each branch of the trie
    if min( currentRow ) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow,
                results, maxCost )

def main():
        TARGET = 'aufgwbe'
        MAX_COST = 1

        # Keep some interesting statistics
        NodeCount = 0
        WordCount = 0

        gb = GraphBuilder()
        trie = gb.trie

        start = time.time()
        results = search( TARGET, MAX_COST, trie )
        end = time.time()

        for result in results:
            print(result)

        print("Search took %g s" % (end - start))

if __name__ == '__main__':
    main()