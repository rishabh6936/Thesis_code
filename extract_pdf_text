import PyPDF2

def text_extractor(path):
    data = []
    with open(path, mode='rb') as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        for i in range(num_pages) :
            page = reader.pages[i]
            data.append(page.extract_text())
    return data
# reader.pages[num_pages]
if __name__ == "__main__":
    path = '/Users/rishabhsingh/Downloads/Bsc_thesis_Rishabh_Singh_final.pdf'
    pypdf = text_extractor(path)
    print('The pdf has {} pages and the data structure is a {} where the index refers to the page number.'.format(len(pypdf), type(pypdf)))
